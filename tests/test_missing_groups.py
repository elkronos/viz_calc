"""Rows whose group is missing: left out and counted by the inferential charts, drawn as "(missing)" by the others."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

import viz_calc as vc
from viz_calc import datasets
from viz_calc._core import NEUTRAL

GONE = [1, 2, 3, 44, 45]  # trial rows given a missing arm (and an outlying score) below


def _as(values, dtype):
    s = pd.Series(values, dtype=object)
    if dtype == "category":
        return s.astype("category")
    return s.astype(dtype)


@pytest.fixture(params=[object, "string", "category"])
def trial_gaps(request):
    t = datasets.trial()
    arm = t["arm"].astype(object)
    arm[GONE] = None
    t["arm"] = _as(arm, request.param)
    t.loc[GONE, "score"] = 500.0
    return t


# --- inferential charts leave the rows out, count them, and take threshold="mean" from the rows drawn ----------


@pytest.mark.parametrize("threshold", ["mean", "median"])
def test_benchmark_bar_threshold_comes_from_the_rows_drawn(trial_gaps, threshold):
    res = vc.benchmark_bar(trial_gaps, x="arm", y="score", threshold=threshold)
    kept = trial_gaps.dropna(subset=["arm"])
    same = vc.benchmark_bar(kept, x="arm", y="score", threshold=threshold)
    expected = kept["score"].mean() if threshold == "mean" else kept["score"].median()
    assert res.info["threshold"] == pytest.approx(expected) == same.info["threshold"]
    assert res.table["n"].sum() == len(kept) == 115
    assert res.table["status"].tolist() == same.table["status"].tolist() == ["indistinguishable"] * 3
    assert res.info["dropped_missing_group"] == 5 and same.info["dropped_missing_group"] == 0
    assert res.axes.lines[-1].get_ydata()[0] == res.info["threshold"]


@pytest.mark.parametrize("threshold", ["mean", "median"])
def test_centered_bar_threshold_comes_from_the_rows_drawn(trial_gaps, threshold):
    res = vc.centered_bar(trial_gaps, x="arm", y="score", threshold=threshold)
    kept = trial_gaps.dropna(subset=["arm"])
    same = vc.centered_bar(kept, x="arm", y="score", threshold=threshold)
    assert res.info["threshold"] == same.info["threshold"]
    assert res.table["n_above"].tolist() == same.table["n_above"].tolist()
    assert res.table["n_above"].sum() >= 50  # the outliers without an arm do not lift the mean above almost every score
    assert res.info["dropped_missing_group"] == 5 and same.info["dropped_missing_group"] == 0
    assert res.axes.get_title(loc="left") == same.axes.get_title(loc="left")


@pytest.mark.parametrize("func", [vc.benchmark_bar, vc.centered_bar])
def test_threshold_mean_leaves_out_groups_not_in_order(func):
    df = pd.DataFrame({"g": list("aabbcc"), "y": [1.0, 3.0, 5.0, 7.0, 100.0, 200.0]})
    res = func(df, x="g", y="y", order=["a", "b"])
    assert res.info["threshold"] == 4.0
    assert res.info["dropped_missing_group"] == 0
    assert func(df, x="g", y="y", order=["a", "b"], threshold="median").info["threshold"] == 4.0


@pytest.mark.filterwarnings("ignore:All-NaN axis encountered:RuntimeWarning")  # Matplotlib 3.7 on a bar of NaN
@pytest.mark.filterwarnings("ignore:Warning. converting a masked element to nan:UserWarning")
@pytest.mark.parametrize("func", [vc.benchmark_bar, vc.centered_bar])
def test_threshold_mean_needs_a_value_in_a_group_drawn(func):
    df = pd.DataFrame({"g": ["a", "a", None, None], "y": [np.nan, np.nan, 1.0, 2.0]})
    with pytest.raises(ValueError, match=r"threshold='mean' needs a value of 'y' in one of the groups drawn"):
        func(df, x="g", y="y")
    res = func(df, x="g", y="y", threshold=1.5)  # a number needs no data
    assert res.table["n"].tolist() == [0] and res.info["dropped_missing_group"] == 2
    with pytest.raises(ValueError, match="column 'y' has no non-missing values"):
        func(df.assign(y=np.nan), x="g", y="y", threshold="nope")


def test_threshold_without_missing_groups_is_unchanged():
    trial = datasets.trial()
    assert vc.benchmark_bar(trial, x="arm", y="score").info["threshold"] == 51.565
    assert vc.centered_bar(trial, x="arm", y="score").info["threshold"] == 51.565
    assert vc.benchmark_bar(trial, x="arm", y="score").info["dropped_missing_group"] == 0


def test_estimation_plot_counts_the_rows_left_out(trial_gaps):
    res = vc.estimation_plot(trial_gaps, x="arm", y="score", n_resamples=200)
    same = vc.estimation_plot(trial_gaps.dropna(subset=["arm"]), x="arm", y="score", n_resamples=200)
    assert res.info["dropped_missing_group"] == 5 and same.info["dropped_missing_group"] == 0
    pd.testing.assert_frame_equal(res.table, same.table)
    pd.testing.assert_frame_equal(res.info["comparisons"], same.info["comparisons"])


def test_raincloud_counts_the_rows_left_out(trial_gaps):
    res = vc.raincloud(trial_gaps, x="arm", y="score")
    assert res.table["n"].tolist() == [37, 40, 38]
    assert res.info["dropped_missing_group"] == 5
    assert vc.raincloud(datasets.trial(), x="arm", y="score").info["dropped_missing_group"] == 0


def test_compare_correlations_counts_the_rows_left_out():
    m = datasets.measurements()
    m["group"] = m["group"].astype(object)
    m.loc[[0, 1, 2], "group"] = None
    res = vc.compare_correlations(m, group="group", columns=["length", "width", "depth"])
    assert res.info["dropped_missing_group"] == 3
    assert sorted(set(res.table["group_a"]) | set(res.table["group_b"])) == ["alpha", "beta", "gamma"]
    clean = vc.compare_correlations(datasets.measurements(), group="group", columns=["length", "width", "depth"])
    assert clean.info["dropped_missing_group"] == 0


# --- descriptive charts draw the rows as a last "(missing)" group, grey where colour marks the group -------------

GREY = to_rgba(NEUTRAL)


def _ticks(labels):
    return [t.get_text() for t in labels]


@pytest.mark.parametrize("stat", ["mean", "sum", "count"])
@pytest.mark.parametrize("sort", ["descending", "ascending", None])
def test_lollipop_draws_a_missing_category_last_in_grey(trial_gaps, stat, sort):
    res = vc.lollipop(trial_gaps, x="arm", y="score", stat=stat, sort=sort)
    table = res.table
    assert table["arm"].isna().tolist() == [False, False, False, True]
    expected = {"mean": 500.0, "sum": 2500.0, "count": 5}[stat]
    assert table["value"].iloc[-1] == expected
    if stat == "count":
        assert table["value"].sum() == len(trial_gaps)
    assert _ticks(res.axes.get_yticklabels())[-1] == "(missing)"
    assert res.axes.get_yticks()[-1] == min(res.axes.get_yticks())  # the last row is at the bottom
    dots = res.axes.collections[-1].get_facecolors()
    assert np.allclose(dots[-1], GREY) and not np.allclose(dots[0], GREY)
    assert to_rgba(res.axes.collections[-2].get_color()[0]) == GREY  # its stem


def test_lollipop_counts_rows_with_a_missing_category():
    df = pd.DataFrame({"k": ["a", None, "b", "a", None]})
    res = vc.lollipop(df, x="k", stat="count", horizontal=False)
    assert res.table["value"].tolist() == [2, 1, 2]
    assert _ticks(res.axes.get_xticklabels()) == ["a", "b", "(missing)"]


def test_lollipop_without_missing_categories_keeps_one_colour():
    res = vc.lollipop(datasets.trial(), x="arm", y="score")
    assert np.allclose(res.axes.collections[-1].get_facecolors(), to_rgba(vc.OKABE_ITO[0]))
    assert "(missing)" not in _ticks(res.axes.get_yticklabels())


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("dtype", [object, "category"])
def test_divergent_bar_sums_a_missing_category_last_in_grey(sort, dtype):
    df = pd.DataFrame({"c": _as(["a", "b", None, "a", None], dtype), "l": [1.0, 2, 30, 4, 5], "r": [5.0, 6, 70, 8, 1]})
    res = vc.divergent_bar(df, category="c", left="l", right="r", sort=sort)
    assert res.table["c"].isna().tolist() == [False, False, True]
    assert res.table[["l", "r"]].iloc[-1].tolist() == [35.0, 71.0]
    assert res.table[["l", "r"]].sum().tolist() == [42.0, 90.0]
    assert _ticks(res.axes.get_yticklabels())[-1] == "(missing)"
    left, right = (c.patches for c in res.axes.containers)
    assert left[-1].get_facecolor() == right[-1].get_facecolor() == GREY
    assert left[0].get_facecolor() != GREY and right[0].get_facecolor() != GREY
    assert [t.get_text() for t in res.axes.get_legend().get_texts()] == ["l", "r"]


@pytest.mark.parametrize("sort", [False, True])
def test_divergent_bar_draws_a_missing_category_last_when_it_comes_first(sort):
    # The missing category appears first in the data and has the smallest total.
    df = pd.DataFrame({"c": [None, "a", "b"], "l": [1.0, 5, 6], "r": [1.0, 5, 6]})
    res = vc.divergent_bar(df, category="c", left="l", right="r", sort=sort)
    assert res.table["c"].isna().tolist() == [False, False, True]
    assert _ticks(res.axes.get_yticklabels()) == ["a", "b", "(missing)"]


@pytest.mark.parametrize("stat", ["mean", "sum", "count"])
def test_a_dictionary_arrow_category_keeps_its_missing_rows_apart(stat):
    # read_parquet(dtype_backend="pyarrow") returns a Categorical as a dictionary-encoded Arrow column.
    pa = pytest.importorskip("pyarrow")
    k = pd.Series(["b", None, "a", "b", None, "z"], dtype=pd.ArrowDtype(pa.dictionary(pa.int32(), pa.string())))
    df = pd.DataFrame({"k": k, "y": [1.0, 2, 3, 4, 5, 6]})
    res = vc.lollipop(df, x="k", y="y", stat=stat, sort=None)
    assert res.table["value"].tolist() == {"mean": [3.0, 2.5, 6.0, 3.5], "sum": [3.0, 5.0, 6.0, 7.0],
                                           "count": [1, 2, 1, 2]}[stat]
    assert _ticks(res.axes.get_yticklabels()) == ["a", "b", "z", "(missing)"]
    bars = vc.divergent_bar(df.assign(r=1.0), category="k", left="y", right="r")
    assert bars.table["y"].tolist() == [5.0, 3.0, 6.0, 7.0] and bars.table["r"].tolist() == [2.0, 1.0, 1.0, 2.0]
    assert _ticks(bars.axes.get_yticklabels()) == ["b", "a", "z", "(missing)"]


@pytest.mark.parametrize("value", [None, "v"])
@pytest.mark.parametrize("role", ["outer", "inner"])
def test_nested_pie_keeps_the_missing_rows_of_a_dictionary_arrow_column(role, value):
    pa = pytest.importorskip("pyarrow")
    k = pd.Series(["b", None, "a", "b", None, "a"], dtype=pd.ArrowDtype(pa.dictionary(pa.int32(), pa.string())))
    df = pd.DataFrame({"k": k, "c": ["u", "u", "w", "w", "u", "u"], "v": 1.0})
    outer, inner = ("k", "c") if role == "outer" else ("c", "k")
    res = vc.nested_pie(df, outer=outer, inner=inner, value=value)
    assert res.table["value"].sum() == 6
    assert res.table["k"].isna().sum() == 1
    assert res.table["percent_of_total"].sum() == pytest.approx(100)


def test_stacked_percentages_draws_a_missing_group_as_the_last_bar():
    s = datasets.survey()
    s["team"] = s["team"].astype(object)
    s.loc[[0, 1, 2, 3], "team"] = None
    res = vc.stacked_percentages(s, group="team", category="remote")
    assert _ticks(res.axes.get_xticklabels())[-1].startswith("(missing)\n(n=4)")
    gone = res.table[res.table["team"].isna()]
    assert gone["n"].sum() == 4 and gone["percent"].sum() == pytest.approx(100)
    assert res.table["n"].sum() == len(s)
    subset = vc.stacked_percentages(s, group="team", category="remote", group_order=["Sales", "Data"])
    assert [t.split("\n")[0] for t in _ticks(subset.axes.get_xticklabels())] == ["Sales", "Data", "(missing)"]


def test_percent_grid_draws_a_missing_facet_as_the_last_panel():
    s = datasets.survey()
    s["team"] = s["team"].astype("category")
    s.loc[[0, 1, 2, 3], "team"] = np.nan
    res = vc.percent_grid(s, column="remote", facet="team", success=True)
    assert res.table["n"].sum() == len(s)
    assert res.table["facet"].isna().tolist() == [False] * 4 + [True]
    titles = [ax.get_title() for ax in res.axes.flat if ax.get_visible()]
    assert titles[-1].startswith("(missing): ") and titles[-1].endswith("n=4")


def test_histogram_draws_a_missing_hue_in_grey_and_a_missing_facet_as_a_panel():
    m = datasets.measurements()
    m["group"] = m["group"].astype(object)
    m.loc[[0, 1, 2], "group"] = None
    res = vc.histogram(m, x="length", hue="group")
    assert res.table["n"].tolist() == [47, 50, 50, 3] and res.table["hue"].isna().tolist()[-1]
    legend = res.axes.flat[0].get_legend()
    assert _ticks(legend.get_texts()) == ["alpha", "beta", "gamma", "(missing)"]
    assert to_rgba(legend.get_lines()[-1].get_color()) == GREY
    facets = vc.histogram(m, x="length", facet="group")
    assert facets.table["n"].tolist() == [47, 50, 50, 3]
    assert [ax.get_title(loc="left") for ax in facets.axes.flat][3] == "group = (missing)"


def test_ridgeplot_draws_a_missing_group_last_in_grey(trial_gaps):
    res = vc.ridgeplot(trial_gaps, x="score", group="arm")
    assert res.table["n"].tolist() == [37, 40, 38, 5] and res.table["arm"].isna().tolist()[-1]
    assert _ticks(res.axes.get_yticklabels())[-1] == "(missing)"
    m = datasets.measurements()
    m["group"] = m["group"].astype(object)
    m.loc[range(0, 150, 10), "group"] = None
    fills = vc.ridgeplot(m, x="length", group="group").axes.collections
    assert len(fills) == 4 and np.allclose(fills[-1].get_facecolors()[0][:3], GREY[:3])


def test_descriptive_charts_without_missing_groups_have_no_missing_level():
    trial, m = datasets.trial(), datasets.measurements()
    assert "(missing)" not in _ticks(vc.ridgeplot(trial, x="score", group="arm").axes.get_yticklabels())
    assert len(vc.histogram(m, x="length", hue="group", facet="group").table) == 9
    assert len(vc.stacked_percentages(datasets.survey(), group="team", category="remote").table) == 8
