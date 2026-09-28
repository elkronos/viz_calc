"""Part-to-whole and timeline charts: missing values, shared and reserved columns, overflow, rates and labels."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import viz_calc as vc
from viz_calc import datasets
from viz_calc._core import NEUTRAL, OKABE_ITO

SALES = pd.DataFrame({"region": ["N", "S", None, None], "product": ["a", "b", "a", None], "v": [10, 10, 40, 40]})


# --- one column passed for two roles --------------------------------------------------------------------------------


def test_gantt_rejects_one_column_for_start_and_end():
    with pytest.raises(ValueError, match="start and end must be different columns"):
        vc.gantt(datasets.projects(), task="task", start="start", end="start")


def test_gantt_rejects_a_date_column_as_task_or_group():
    proj = datasets.projects()
    with pytest.raises(ValueError, match="task and start must be different columns"):
        vc.gantt(proj, task="start", start="start", end="end")
    with pytest.raises(ValueError, match="group and end must be different columns"):
        vc.gantt(proj, task="task", start="start", end="end", group="end")


def test_gantt_may_colour_each_task_by_its_own_name():
    res = vc.gantt(datasets.projects(), task="task", start="start", end="end", group="task")
    assert list(res.table.columns) == ["task", "start", "end", "duration_days"]


@pytest.mark.parametrize("roles, message", [
    (("task", "start", "start", "start", "end"), "start and end"),
    (("end", "start", "end", "start", "end"), "label and end"),
    (("task", "start", "end", "end", "end"), "inner_start and inner_end"),
    (("task", "start", "end", "end", "x"), "end and inner_start"),
    (("task", "start", "end", "x", "start"), "start and inner_end"),
])
def test_duration_plot_rejects_one_column_for_two_roles(roles, message):
    proj = datasets.projects().assign(x=lambda d: d["end"])
    with pytest.raises(ValueError, match=f"{message} must be different columns"):
        vc.duration_plot(proj, *roles)


def test_duration_plot_inner_window_may_reuse_the_whole_period():
    proj = datasets.projects()
    res = vc.duration_plot(proj, label="task", start="start", end="end", inner_start="start", inner_end="end")
    assert list(res.table.columns) == ["task", "start", "end", "outer_days", "inner_days", "inner_share"]
    assert (res.table["inner_days"] == res.table["outer_days"]).all()
    assert (res.table["inner_share"].dropna() == 1).all()


# --- missing categories are kept -------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [object, "category", "string"])
def test_nested_pie_keeps_rows_with_a_missing_category(dtype):
    data = SALES.astype({"region": dtype, "product": dtype})
    res = vc.nested_pie(data, "region", "product", "v")
    t = res.table
    assert t["value"].sum() == 100
    assert t["percent_of_total"].tolist() == [10.0, 10.0, 40.0, 40.0]
    assert t["region"].isna().tolist() == [False, False, True, True]  # the missing parent comes last
    assert t["percent_of_parent"].tolist() == [100.0, 100.0, 50.0, 50.0]
    texts = [x.get_text() for x in res.axes.texts]
    assert texts == ["N", "S", "(missing)", "a", "b", "a", "(missing)"]
    inner_ring = res.axes.patches[3:]  # three parent wedges, then the children
    assert to_hex(inner_ring[-1].get_facecolor()) == to_hex(NEUTRAL)
    assert to_hex(res.axes.patches[2].get_facecolor()) == to_hex(NEUTRAL)


def test_nested_pie_missing_inner_under_a_known_parent():
    data = pd.DataFrame({"o": ["x", "x", "y"], "i": ["p", None, "q"]})
    t = vc.nested_pie(data, "o", "i").table
    assert t["value"].tolist() == [1, 1, 1]
    assert t["percent_of_parent"].tolist() == [50.0, 50.0, 100.0]


@pytest.mark.parametrize("dtype", [object, "category", "string"])
def test_waffle_keeps_rows_with_a_missing_category(dtype):
    data = SALES.astype({"region": dtype})
    res = vc.waffle(data, "region", "v")
    t = res.table
    assert t["value"].tolist() == [10, 10, 80]
    assert t["percent"].tolist() == [10.0, 10.0, 80.0]
    assert t["tiles"].tolist() == [10, 10, 80]
    assert t["region"].isna().tolist() == [False, False, True]
    legend = [x.get_text() for x in res.axes.get_legend().get_texts()]
    assert legend[-1] == "(missing): 80 (80.0%)"
    assert to_hex(res.axes.patches[-1].get_facecolor()) == to_hex(NEUTRAL)


def test_waffle_counts_and_order_include_missing_category():
    t = vc.waffle(SALES, "region").table
    assert t["value"].tolist() == [1, 1, 2]
    t = vc.waffle(SALES, "region", "v", order=["S", "N"]).table
    assert t["region"].tolist()[:2] == ["S", "N"] and t["percent"].tolist() == [10.0, 10.0, 80.0]


def test_waffle_positive_total_only_in_rows_with_missing_category():
    data = pd.DataFrame({"c": ["a", None], "v": [0, 5]})
    assert vc.waffle(data, "c", "v").table["tiles"].tolist() == [0, 100]


# --- bullet ----------------------------------------------------------------------------------------------------------


def test_bullet_integer_band_labels_are_columns():
    data = pd.DataFrame({0: ["a", "b"], 1: [3.0, 5.0], 2: [4.0, 6.0], 3: [8.0, 9.0]})
    res = vc.bullet(data, label=0, value=1, bands=[2, 3])
    bands = [(float(p.get_x()), float(p.get_x() + p.get_width())) for p in res.axes[0].patches[:2]]
    assert bands == [(0.0, 4.0), (4.0, 8.0)]


def test_bullet_numeric_limits_that_are_not_all_columns_stay_limits():
    data = pd.DataFrame({0: ["a"], 1: [3.0], 2: [4.0]})
    res = vc.bullet(data, label=0, value=1, bands=[2, 5])
    bands = [(float(p.get_x()), float(p.get_x() + p.get_width())) for p in res.axes[0].patches[:2]]
    assert bands == [(0.0, 2.0), (2.0, 5.0)]


@pytest.mark.parametrize("name", ["value", "target", "pct_of_target"])
def test_bullet_rejects_label_named_like_an_output_column(name):
    data = pd.DataFrame({name: ["Revenue", "Profit"], "actual": [270, 22], "goal": [250, 26]})
    with pytest.raises(ValueError, match="rename the label column"):
        vc.bullet(data, label=name, value="actual", target="goal")


def test_bullet_label_named_target_without_target_is_kept():
    data = pd.DataFrame({"target": ["Revenue", "Profit"], "actual": [270, 22]})
    t = vc.bullet(data, label="target", value="actual").table
    assert t["target"].tolist() == ["Revenue", "Profit"]


# --- funnel and period_bars ------------------------------------------------------------------------------------------


def test_funnel_undefined_rates_are_not_printed_as_nan():
    res = vc.funnel(pd.DataFrame({"stage": ["visit", "signup", "buy"], "n": [0, 0, 0]}), "stage", "n")
    texts = [x.get_text() for x in res.axes.texts]
    assert not any("nan" in x for x in texts)
    assert texts[:3] == ["0  (–)", "0  (–)", "↓ – of previous"]
    assert res.table["pct_of_first"].isna().all()


def test_funnel_rate_after_a_zero_stage():
    res = vc.funnel(pd.DataFrame({"stage": list("abc"), "n": [10, 0, 0]}), "stage", "n")
    texts = [x.get_text() for x in res.axes.texts]
    assert "↓ 0% of previous" in texts and "↓ – of previous" in texts


def test_period_bars_int64_sum_does_not_wrap():
    data = pd.DataFrame({"d": ["2024-01-01", "2024-01-02", "2024-02-01"],
                         "v": np.array([5 * 10**18, 5 * 10**18, 3], dtype=np.int64)})
    res = vc.period_bars(data, "d", "v")
    assert res.table["value"].tolist() == [1e19, 3.0]
    assert res.axes.patches[0].get_height() == 1e19


def test_period_bars_small_int_sums_stay_exact_integers():
    data = pd.DataFrame({"d": ["2024-01-01", "2024-01-02"], "v": np.array([2**53, 1], dtype=np.int64)})
    assert vc.period_bars(data, "d", "v").table["value"].tolist() == [2**53 + 1]


# --- upset: missing members ------------------------------------------------------------------------------------------


def test_upset_mapping_does_not_count_missing_values_as_members():
    r = vc.upset({"A": ["u1", "u2", None], "B": ["u2", "u3", pd.NA]})
    assert r.info["set_sizes"].to_dict() == {"A": 2, "B": 2}
    assert sorted(r.table["size"]) == [1, 1, 1]


def test_upset_mapping_from_columns_with_blanks_matches_the_dataframe_path():
    df = pd.DataFrame({"A": ["u1", "u2", np.nan], "B": ["u2", "u3", np.nan]})
    r = vc.upset({c: df[c] for c in df})  # one shared NaN object must not become a common member
    assert r.table.loc[r.table["A"] & r.table["B"], "size"].tolist() == [1]
    assert r.info["set_sizes"].to_dict() == {"A": 2, "B": 2}


# --- donut_grid: columns labelled False/True -------------------------------------------------------------------------


def test_donut_grid_selects_boolean_column_labels_as_columns():
    wide = pd.DataFrame({False: [1.0, 3.0], True: [3.0, 1.0]}, index=["x", "y"])
    r = vc.donut_grid(wide)
    assert r.table["category"].tolist() == [False, False, True, True]
    assert r.table["percent"].tolist() == [25.0, 75.0, 75.0, 25.0]
    assert vc.donut_grid(wide, columns=[True]).table["percent"].tolist() == [100.0, 100.0]


def test_donut_grid_reports_missing_cells_in_boolean_labelled_columns():
    wide = pd.DataFrame({False: [1.0, np.nan], True: [3.0, 1.0]}, index=["x", "y"])
    with pytest.raises(ValueError, match=r"row\(s\) \['y'\] have missing values"):
        vc.donut_grid(wide)


# --- percent_grid ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("success", [pd.Timestamp("2024-01-02"), np.datetime64("2024-01-02"),
                                     np.datetime64("2024-01-02T00:00:00.000", "ms")])
def test_percent_grid_accepts_a_datetime_success_value(success):
    df = pd.DataFrame({"shipped": pd.to_datetime(["2024-01-01", "2024-01-02"] * 5)})
    r = vc.percent_grid(df, "shipped", success=success)
    assert r.table[["n", "successes", "percent"]].to_dict("records") == [{"n": 10, "successes": 5, "percent": 50.0}]


def test_percent_grid_still_rejects_a_datetime_that_does_not_occur():
    df = pd.DataFrame({"shipped": pd.to_datetime(["2024-01-01", "2024-01-02"] * 5)})
    with pytest.raises(ValueError, match="does not occur"):
        vc.percent_grid(df, "shipped", success=pd.Timestamp("2024-01-03"))


@pytest.mark.parametrize(("k", "n", "dots"), [(1, 200, 1), (5, 200, 3), (7, 200, 4), (1, 8, 13), (3, 8, 38),
                                              (1, 3, 33), (2, 3, 67), (0, 7, 0), (7, 7, 100), (1, 201, 0)])
def test_percent_grid_dots_round_halves_up(k, n, dots):
    r = vc.percent_grid(pd.DataFrame({"ok": [1] * k + [0] * (n - k)}), "ok")
    fc = r.axes.flat[0].collections[0].get_facecolors()
    assert sum(to_hex(c) == to_hex(OKABE_ITO[0]) for c in fc) == dots


# --- bullet: missing band limits -------------------------------------------------------------------------------------


def _bands(ax):
    return [(float(p.get_x()), float(p.get_width())) for p in ax.patches[:-1]]  # the last patch is the value bar


def test_bullet_missing_first_band_limit_keeps_the_other_bands():
    df = pd.DataFrame({"kpi": ["Profit"], "value": [20], "target": [25], "poor": [np.nan], "ok": [20], "good": [30]})
    r = vc.bullet(df, "kpi", "value", "target", bands=["poor", "ok", "good"])
    assert _bands(r.axes[0]) == [(0.0, 20.0), (20.0, 10.0)]


def test_bullet_missing_middle_band_limit_keeps_the_later_band():
    df = pd.DataFrame({"kpi": ["a", "b"], "value": [5.0, 5.0], "poor": [10, 10], "ok": [np.nan, 20],
                       "good": [30, 30]})
    r = vc.bullet(df, "kpi", "value", bands=["poor", "ok", "good"])
    assert _bands(r.axes[0]) == [(0.0, 10.0), (10.0, 20.0)]
    assert _bands(r.axes[1]) == [(0.0, 10.0), (10.0, 10.0), (20.0, 10.0)]


def test_bullet_nullable_band_column_with_pd_na():
    df = pd.DataFrame({"kpi": ["a"], "value": [5.0], "poor": pd.array([pd.NA], dtype="Int64"), "good": [30]})
    assert _bands(vc.bullet(df, "kpi", "value", bands=["poor", "good"]).axes[0]) == [(0.0, 30.0)]


# --- funnel labels ---------------------------------------------------------------------------------------------------


def _funnel_texts(values):
    r = vc.funnel(pd.DataFrame({"stage": [f"s{i}" for i in range(len(values))], "n": values}), "stage", "n")
    return [t.get_text() for t in r.axes.texts]


def test_funnel_small_overall_rate_is_not_shown_as_zero():
    texts = _funnel_texts([100000, 2400, 450])
    assert "450  (0.45%)" in texts
    assert "2,400  (2%)" in texts  # whole percent otherwise


def test_funnel_rate_just_below_100_is_not_shown_as_100():
    texts = _funnel_texts([1000, 996])
    assert "996  (99.6%)" in texts
    assert "↓ 99.6% of previous" in texts
    assert "1,000  (100%)" in texts


def test_funnel_rate_just_above_100_is_not_shown_as_100():
    assert "↓ 100.2% of previous" in _funnel_texts([1000, 1002])


def test_funnel_zero_rate_still_reads_zero():
    assert "0  (0%)" in _funnel_texts([10, 0])


# --- circular_bar, waffle, animated_bubble ---------------------------------------------------------------------------


def test_circular_bar_all_zero_values_title_max_zero():
    r = vc.circular_bar(pd.DataFrame({"l": ["a", "b"], "v": [0, 0]}), "l", "v")
    assert r.axes.get_title() == "v (bar length relative to max = 0)"
    assert "max = 5)" in vc.circular_bar(pd.DataFrame({"l": ["a", "b"], "v": [0, 5]}), "l", "v").axes.get_title()


def test_waffle_int64_sums_do_not_wrap():
    df = pd.DataFrame({"c": ["a", "a", "b"], "v": np.array([2**62, 2**62, 1], dtype=np.int64)})
    r = vc.waffle(df, "c", "v")
    assert r.table["tiles"].tolist() == [100, 0]
    assert r.table["value"].iloc[0] == 2.0**63


def test_waffle_int64_sums_do_not_wrap_for_rows_with_missing_category():
    df = pd.DataFrame({"c": ["a", None, None], "v": np.array([1, 2**62, 2**62], dtype=np.int64)})
    r = vc.waffle(df, "c", "v")
    assert r.table["tiles"].tolist() == [0, 100]
    assert r.table["value"].iloc[1] == 2.0**63


def test_waffle_small_int_sums_stay_integers():
    r = vc.waffle(pd.DataFrame({"c": ["a", "a", "b"], "v": [2, 3, 5]}), "c", "v")
    assert r.table["value"].tolist() == [5, 5]
    assert pd.api.types.is_integer_dtype(r.table["value"])


def test_animated_bubble_mixed_number_and_text_times_raise_clearly():
    d = pd.DataFrame({"t": [2019, "2020", 2021], "x": [1.0, 2, 3], "y": [1.0, 2, 3], "s": [1.0, 1, 1]})
    with pytest.raises(ValueError, match=r"column 't' mixes values that cannot be ordered \(int, str\)"):
        vc.animated_bubble(d, "t", "x", "y", "s")


# --- upset ----------------------------------------------------------------------------------------------------------

def test_upset_boolean_set_names_give_correct_degrees():
    two = pd.DataFrame({False: [True, False], True: [True, True]})  # intersections {False, True} and {True}
    table = vc.upset(two).table
    degrees = {tuple(bool(v) for v in row[[0, 1]]): row["degree"] for _, row in table.iterrows()}
    assert degrees == {(True, True): 2, (False, True): 1}


def test_upset_boolean_set_names_with_three_intersections():
    three = pd.DataFrame({False: [True, False, True, True], True: [True, True, False, True]})
    res = vc.upset(three, sort_by="degree")
    assert res.table["degree"].tolist() == [1, 1, 2]
    assert res.table["size"].tolist() == [1, 1, 2]
    # the matrix marks each intersection's members: two dots for {False, True}, one for the others
    members = [len(c.get_offsets()) for c in res.axes[1].collections[1::2]]
    assert members == [1, 1, 2]


# --- percent_grid ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("level, text", [(0.999, "99.9% CI"), (0.975, "97.5% CI"), (0.95, "95% CI"),
                                         (0.8, "80% CI")])
def test_percent_grid_level_label_is_not_rounded(level, text):
    d = pd.DataFrame({"s": [1, 0, 1, 1, 1, 0]})
    title = vc.percent_grid(d, "s", level=level).axes.flat[0].get_title()
    assert title.splitlines()[1].startswith(text + " ")


# --- animated_bubble ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("role", ["x", "y", "size"])
def test_animated_bubble_time_in_another_role_raises(role):
    t = datasets.trial().assign(yr=lambda d: d.index % 2)
    kwargs = {"x": "score", "y": "baseline", "size": "score", role: "yr"}
    with pytest.raises(ValueError, match=f"time and {role} must be different columns; both are 'yr'"):
        vc.animated_bubble(t, time="yr", **kwargs)


def test_animated_bubble_shared_column_appears_once_in_table():
    t = datasets.trial().assign(yr=lambda d: d.index % 2)
    table = vc.animated_bubble(t, time="yr", x="score", y="baseline", size="score").table
    assert table.columns.tolist() == ["yr", "score", "baseline"]


# --- nested_pie -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("renamed, outer, inner", [("arm", "value", "site"), ("site", "arm", "value"),
                                                   ("arm", "percent_of_total", "site")])
def test_nested_pie_category_named_like_output_column_raises(renamed, outer, inner):
    new = outer if renamed == "arm" else inner
    data = datasets.trial().rename(columns={renamed: new})
    with pytest.raises(ValueError, match=rf"rename column\(s\) \['{new}'\]"):
        vc.nested_pie(data, outer, inner)


def test_nested_pie_value_column_named_value_still_works():
    data = datasets.trial().assign(value=1.0)
    table = vc.nested_pie(data, "arm", "site", value="value").table
    assert table["value"].sum() == len(data)


# --- all-missing group column ---------------------------------------------------------------------------------------

def test_circular_bar_all_missing_group_draws_one_missing_cluster():
    w = pd.DataFrame({"lab": list("abcd"), "v": [4.0, 3, 2, 1], "grp": [None] * 4})
    res = vc.circular_bar(w, "lab", "v", group="grp")
    assert res.table["lab"].tolist() == list("abcd")
    texts = [t.get_text() for t in res.axes.texts]
    assert "(missing)" in texts


def test_gantt_all_missing_group_draws_every_task_as_missing():
    projects = datasets.projects().assign(team=None)
    res = vc.gantt(projects, "task", "start", "end", group="team")
    assert len(res.table) == len(projects)
    assert [t.get_text() for t in res.axes.get_legend().get_texts()] == ["(missing)"]


def test_waffle_rejects_a_category_named_like_its_output():
    with pytest.raises(ValueError, match="rename"):
        vc.waffle(pd.DataFrame({"percent": ["a", "b", "a"]}), category="percent")


def test_percent_grid_titles_show_the_exact_level():
    df = pd.DataFrame({"ok": [1, 0, 1, 1]})
    res = vc.percent_grid(df, column="ok", level=0.975)
    assert "97.5% CI" in res.axes.flat[0].get_title() if hasattr(res.axes, "flat") else "97.5% CI" in res.axes.get_title()
