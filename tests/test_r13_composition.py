"""Regressions for the part-to-whole charts found in review round 13."""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets


def _render(res):
    res.figure.savefig(io.BytesIO(), format="png")  # some failures only appear when the figure is drawn


# --- bullet: ax= ------------------------------------------------------------------------------------------------

KPI = pd.DataFrame({"kpi": ["Revenue"], "value": [270], "target": [250], "poor": [150], "ok": [225], "good": [300]})


def test_bullet_draws_a_single_row_into_the_given_ax():
    fig, (left, right) = plt.subplots(1, 2)
    position = right.get_position().bounds
    res = vc.bullet(KPI, label="kpi", value="value", target="target", bands=["poor", "ok", "good"], ax=right)
    assert res.figure is fig and res.axes[0] is right and len(res.axes) == 1
    assert len(right.patches) == 4 and len(right.lines) == 1  # three bands, the measure and the target
    assert plt.get_fignums() == [fig.number]
    assert right.get_position().bounds == position  # the caller's layout is left alone
    assert res.table["pct_of_target"].iloc[0] == pytest.approx(108.0)


def test_bullet_with_ax_and_several_rows_raises_before_drawing():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="single row, got 2 rows"):
        vc.bullet(pd.concat([KPI, KPI]), label="kpi", value="value", ax=ax)
    assert plt.get_fignums() == [fig.number] and not ax.patches


# --- nested_pie ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{"value": "score"}, {}])
def test_nested_pie_categorical_outer_matches_strings(kwargs):
    trial = datasets.trial().iloc[5:105]  # 'arm' is an ordered Categorical; unequal group sizes
    cat = vc.nested_pie(trial, outer="arm", inner="site", **kwargs).table
    txt = vc.nested_pie(trial.assign(arm=trial["arm"].astype(str)), outer="arm", inner="site", **kwargs).table
    assert cat["percent_of_parent"].tolist() == pytest.approx(txt["percent_of_parent"].tolist())
    assert cat.groupby("arm", observed=True)["percent_of_parent"].sum().tolist() == pytest.approx([100.0] * 3)


def test_nested_pie_rejects_negative_values_and_a_zero_total():
    df = pd.DataFrame({"o": ["x", "x", "y"], "i": ["p", "q", "r"], "v": [5, -1, 3]})
    with pytest.raises(ValueError, match="non-negative"):
        vc.nested_pie(df, "o", "i", "v")  # summing first would net the -1 against the 5
    with pytest.raises(ValueError, match="positive total"):
        vc.nested_pie(df.assign(v=0), "o", "i", "v")
    with pytest.raises(ValueError, match="positive total"):
        vc.nested_pie(df.assign(v=np.nan), "o", "i", "v")


# --- stacked_percentages: repeated index labels -------------------------------------------------------------------


def test_stacked_percentages_with_repeated_index_labels():
    both = pd.concat([datasets.survey(seed=2), datasets.survey(seed=7)])  # index 0..299 appears twice
    res = vc.stacked_percentages(both, group="team", category="Tools are adequate")
    ref = vc.stacked_percentages(both.reset_index(drop=True), group="team", category="Tools are adequate")
    pd.testing.assert_frame_equal(res.table, ref.table)
    trials = pd.concat([datasets.trial(), datasets.trial()])  # Categorical group column
    assert vc.stacked_percentages(trials, group="arm", category="site").table["n"].sum() == 2 * len(datasets.trial())


# --- circular_bar -----------------------------------------------------------------------------------------------

ITEMS = pd.DataFrame({"lab": list("abcdef"), "v": [5, 3, 8, 1, 2, 7], "g": ["x", "x", "y", "y", "y", "x"]})


def test_circular_bar_needs_a_polar_axes_and_checks_before_drawing():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="polar Axes"):
        vc.circular_bar(ITEMS, label="lab", value="v", ax=ax)
    assert not ax.patches and not ax.texts and not ax.lines
    polar = plt.figure().add_subplot(projection="polar")
    res = vc.circular_bar(ITEMS, label="lab", value="v", ax=polar)
    assert res.axes is polar and len(polar.patches) == 6


@pytest.mark.filterwarnings("error")  # recoding a Categorical with unused categories is deprecated in pandas 3
@pytest.mark.parametrize("convert", [lambda d: d, lambda d: d.convert_dtypes(),
                                     lambda d: d.assign(g=pd.Categorical(d["g"], categories=["y", "x", "unused"]))],
                         ids=["object", "nullable", "cat"])
def test_circular_bar_keeps_rows_with_a_missing_group(convert):
    d = convert(ITEMS.assign(g=["x", "x", None, "y", "y", "x"]))
    res = vc.circular_bar(d, label="lab", value="v", group="g")
    assert len(res.table) == 6 and len(res.axes.patches) == 6
    assert res.table["lab"].iloc[-1] == "c" and res.table["g"].isna().iloc[-1]  # the grey cluster comes last
    assert "(missing)" in [t.get_text() for t in res.axes.texts]
    assert res.axes.get_title().endswith("max = 8)")
    _render(res)


def test_circular_bar_group_with_nullable_dtypes():
    res = vc.circular_bar(ITEMS.convert_dtypes(), label="lab", value="v", group="g")  # string / Int64 columns
    ref = vc.circular_bar(ITEMS, label="lab", value="v", group="g")
    assert res.table["lab"].tolist() == ref.table["lab"].tolist()
    flags = ITEMS.assign(g=pd.array([True, False, True, None, False, True], dtype="boolean"))
    assert len(vc.circular_bar(flags, label="lab", value="v", group="g").table) == 6


@pytest.mark.parametrize("values", [[5, np.nan, 8, 1, 2, 7], pd.array([5, None, 8, 1, 2, 7], dtype="Int64")],
                         ids=["float", "Int64"])
def test_circular_bar_missing_value_leaves_an_empty_slot(values):
    res = vc.circular_bar(ITEMS.assign(v=values), label="lab", value="v", group="g")
    assert len(res.axes.patches) == 5 and len(res.table) == 6
    assert "b" in [t.get_text() for t in res.axes.texts]
    _render(res)  # a NaN polar bar used to fail when the figure was drawn


def test_circular_bar_rejects_a_negative_gap():
    with pytest.raises(ValueError, match="gap must be an integer >= 0"):
        vc.circular_bar(ITEMS, label="lab", value="v", group="g", gap=-3)


# --- donut_grid ---------------------------------------------------------------------------------------------------

WIDE = pd.DataFrame({"a": [3, 5, 2], "b": [1, 2, 3], "c": [4, 0, 1]})


@pytest.mark.parametrize("columns", [WIDE.columns[:2], np.array(["a", "b"]), ("a", "b"), pd.Series(["a", "b"])],
                         ids=["Index", "ndarray", "tuple", "Series"])
def test_donut_grid_accepts_any_sequence_of_columns(columns):
    res = vc.donut_grid(WIDE, columns=columns)
    assert res.table["category"].unique().tolist() == ["a", "b"]


@pytest.mark.parametrize("wide", [WIDE.astype(float).where(WIDE != 2), WIDE.astype("Int64").where(WIDE != 2)],
                         ids=["float", "Int64"])
def test_donut_grid_rejects_missing_values_clearly(wide):
    with pytest.raises(ValueError, match=r"row\(s\) \[1, 2\] have missing values"):
        vc.donut_grid(wide)
    assert vc.donut_grid(wide.fillna(0)).table["percent"].notna().all()


def test_donut_grid_output_names_do_not_collide_with_columns():
    t = vc.donut_grid(pd.DataFrame({"row": [1, 2], "percent": [3, 2]})).table
    assert t.columns.tolist() == ["row", "category", "percent"]
    assert t["category"].tolist() == ["row", "row", "percent", "percent"]
    assert t["percent"].tolist() == pytest.approx([25.0, 50.0, 75.0, 50.0])


def test_donut_grid_needs_a_column():
    with pytest.raises(ValueError, match="at least one category column"):
        vc.donut_grid(pd.DataFrame({"name": ["x", "y"]}))


# --- funnel -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("values", [[1000.0, np.nan, 50.0], pd.array([1000, None, 50], dtype="Int64")],
                         ids=["float", "Int64"])
def test_funnel_rejects_a_missing_stage_clearly(values):
    with pytest.raises(ValueError, match=r"missing for stage\(s\) \['cart'\]"):
        vc.funnel(pd.DataFrame({"stage": ["visit", "cart", "buy"], "n": values}), "stage", "n")


def test_funnel_rejects_negative_values_and_output_column_names():
    with pytest.raises(ValueError, match="funnel values must be non-negative"):
        vc.funnel(pd.DataFrame({"s": ["visit", "cart"], "n": [100, -5]}), "s", "n")
    with pytest.raises(ValueError, match=r"rename column\(s\) \['drop_off'\]"):
        vc.funnel(pd.DataFrame({"s": ["a", "b"], "drop_off": [100, 50]}), "s", "drop_off")


def test_funnel_nullable_integers_without_gaps():
    t = vc.funnel(pd.DataFrame({"s": list("abc"), "n": pd.array([100, 50, 10], dtype="Int64")}), "s", "n").table
    assert t["pct_of_first"].tolist() == pytest.approx([100.0, 50.0, 10.0])


# --- waterfall ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("values", [[10.0, 5.0, np.nan, 3.0], pd.array([10, 5, None, 3], dtype="Int64")],
                         ids=["float", "Int64"])
def test_waterfall_rejects_missing_values_instead_of_a_nan_total(values):
    with pytest.raises(ValueError, match=r"missing for \['c'\]"):
        vc.waterfall(pd.DataFrame({"lab": list("abcd"), "v": values}), "lab", "v")


def test_waterfall_values_are_checked_before_drawing():
    with pytest.raises(ValueError, match=r"values_are must be one of \['changes', 'levels'\], got 'level'"):
        vc.waterfall(pd.DataFrame({"l": ["a"], "v": [1]}), "l", "v", values_are="level")
    assert not plt.get_fignums()


# --- bullet: missing values ---------------------------------------------------------------------------------------


def test_bullet_nullable_missing_values_are_left_out():
    df = pd.DataFrame({"k": ["a", "b"], "v": pd.array([1, None], dtype="Int64"),
                       "t": pd.array([None, 2], dtype="Int64")})
    res = vc.bullet(df, "k", "v", "t", bands=[1, 2])  # pd.NA used to break float() on pandas 2.0
    assert res.table["pct_of_target"].isna().all()
    _render(res)


# --- upset --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sets", [{"A": {1, 2, 3}}, pd.DataFrame({"A": [1, 0, 1, 1]})], ids=["mapping", "frame"])
def test_upset_draws_a_single_set(sets):
    res = vc.upset(sets)
    assert res.table.columns.tolist() == ["A", "size", "degree"] and res.table["size"].tolist() == [3]
    _render(res)


@pytest.mark.parametrize("sets", [{"degree": {1, 2}, "certificate": {2, 3}},
                                  pd.DataFrame({"size": [1, 0], "B": [0, 1]})], ids=["degree", "size"])
def test_upset_rejects_set_names_that_are_output_columns(sets):
    with pytest.raises(ValueError, match=r"rename set\(s\)"):
        vc.upset(sets)


def test_upset_needs_a_set():
    with pytest.raises(ValueError, match="at least one set"):
        vc.upset({})


# --- waffle -------------------------------------------------------------------------------------------------------


def test_waffle_drops_unused_categories_when_counting_too():
    d = pd.DataFrame({"c": pd.Categorical(list("aabc"), categories=list("abcz")), "v": [1, 2, 3, 4]})
    counted = vc.waffle(d, "c").table
    summed = vc.waffle(d, "c", "v").table
    assert set(counted["c"]) == set(summed["c"]) == {"a", "b", "c"}
    listed = vc.waffle(d, "c", order=list("abcz")).table  # listed in order=, an unused level is drawn as zero
    assert listed["c"].tolist() == list("abcz") and listed["tiles"].iloc[-1] == 0


def test_waffle_rejects_a_negative_row_even_if_the_total_is_positive():
    with pytest.raises(ValueError, match="waffle values must be non-negative"):
        vc.waffle(pd.DataFrame({"c": ["a", "a", "b"], "v": [10, -8, 5]}), "c", "v")


@pytest.mark.parametrize("kwargs", [{"rows": 0}, {"columns": 0}, {"rows": 2.5}, {"rows": True}])
def test_waffle_grid_size_is_validated(kwargs):
    with pytest.raises(ValueError, match=f"{next(iter(kwargs))} must be an integer >= 1"):
        vc.waffle(datasets.trial(), category="arm", **kwargs)


# --- percent_grid and col_wrap ------------------------------------------------------------------------------------


def test_percent_grid_titles_an_empty_group_as_no_data():
    res = vc.percent_grid(pd.DataFrame({"y": [1, 0, np.nan], "f": ["A", "A", "B"]}), "y", facet="f")
    titles = [ax.get_title() for ax in res.axes.flat if ax.get_visible()]
    assert titles[1] == "B: no data (n=0)" and "nan" not in titles[0]
    assert res.table["percent"].isna().tolist() == [False, True]


@pytest.mark.parametrize("call", [
    lambda: vc.percent_grid(datasets.trial(), column="improved", facet="arm", col_wrap=0),
    lambda: vc.donut_grid(WIDE, col_wrap=0),
], ids=["percent_grid", "donut_grid"])
def test_col_wrap_is_validated(call):
    with pytest.raises(ValueError, match="col_wrap must be an integer >= 1"):
        call()
