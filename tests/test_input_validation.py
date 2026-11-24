"""Input checks shared by the charts: column roles, infinite and missing values, read-only columns, sums."""

import contextlib
import re
from decimal import Decimal
from fractions import Fraction
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets
from viz_calc import stats as st
from viz_calc._core import check_finite, level_mask, safe_sum

SMALL = pd.DataFrame({"k": list("abcd"), "o": list("xxyy"), "a": [3.0, 5, 2, 8], "b": [4.0, 6, 3, 9]})


@pytest.fixture(scope="module")
def trial():
    return datasets.trial()


# --- one column passed for two roles --------------------------------------------------------------------------------


@pytest.mark.parametrize("call, message", [
    (lambda t: vc.estimation_plot(t, x="score", y="score"), "x and y must be different columns; both are 'score'"),
    (lambda t: vc.benchmark_bar(t, x="score", y="score"), "x and y must be different columns"),
    (lambda t: vc.lollipop(t, x="score", y="score"), "x and y must be different columns"),
    (lambda t: vc.lollipop(t, x="score", y="score", stat="count"), "x and y must be different columns"),
    (lambda t: vc.centered_bar(t, x="score", y="score"), "x and y must be different columns"),
    (lambda t: vc.raincloud(t, x="score", y="score"), "x and y must be different columns"),
    (lambda t: vc.histogram(t, x="score", hue="score"), "x and hue must be different columns"),
    (lambda t: vc.histogram(t, x="score", facet="score"), "x and facet must be different columns"),
    (lambda t: vc.ridgeplot(t, x="score", group="score"), "x and group must be different columns"),
    (lambda t: vc.percent_grid(t, column="improved", facet="improved"), "column and facet must be different columns"),
    (lambda t: vc.profile_boxes(t, categorical=["arm", "score"], numeric=["baseline", "score"]),
     r"categorical and numeric must be different columns; both name \['score'\]"),
    (lambda t: vc.pca_plot(t, features=["baseline", "score", "improved"], group="improved"),
     "group column 'improved' cannot also be one of features"),
    (lambda t: vc.waffle(SMALL, category="a", value="a"), "category and value must be different columns; both are 'a'"),
    (lambda t: vc.nested_pie(SMALL, outer="a", inner="k", value="a"), "outer and value must be different columns"),
    (lambda t: vc.nested_pie(SMALL, outer="o", inner="a", value="a"), "inner and value must be different columns"),
    (lambda t: vc.nested_pie(SMALL, outer="o", inner="o", value="a"), "outer and inner must be different columns"),
    (lambda t: vc.circular_bar(SMALL, label="k", value="a", group="a"), "value and group must be different columns"),
    (lambda t: vc.bullet(SMALL, label="k", value="a", bands=["b", "b"]), r"bands repeats column\(s\): \['b'\]"),
], ids=["estimation_plot", "benchmark_bar", "lollipop", "lollipop count", "centered_bar", "raincloud",
        "histogram hue", "histogram facet", "ridgeplot", "percent_grid", "profile_boxes", "pca_plot", "waffle",
        "nested_pie outer", "nested_pie inner", "nested_pie outer=inner", "circular_bar", "bullet bands"])
def test_one_column_for_two_roles_raises_before_drawing(trial, call, message):
    with pytest.raises(ValueError, match=message):
        call(trial)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("call, rows", [
    (lambda t: vc.histogram(t, x="score", hue="arm", facet="arm"), 9),  # 3 panels x 3 hues, each panel one colour
    (lambda t: vc.profile_boxes(t, categorical=["arm", "score"]), 3),  # score is also picked as numeric: no self-pair
    (lambda t: vc.bullet(SMALL, label="a", value="a", target="a"), 4),
    (lambda t: vc.waterfall(SMALL, label="a", value="a"), 5),
    (lambda t: vc.circular_bar(SMALL, label="a", value="a", group="k"), 4),
    (lambda t: vc.bullet(SMALL, label="k", value="a", bands=[5, 5]), 4),  # repeated numeric limits are not columns
], ids=["histogram hue=facet", "profile_boxes", "bullet", "waterfall", "circular_bar", "bullet numeric bands"])
def test_roles_that_may_share_a_column(trial, call, rows):
    assert len(call(trial).table) == rows


# --- infinite values in numeric roles ------------------------------------------------------------------------------

NUMBERS = pd.DataFrame({
    "g": list("abc") * 4,
    "k": [f"r{i:02d}" for i in range(12)],
    "v": [3.0, 5.0, 2.0, 8.0, 4.0, 6.0, 7.5, 1.0, 9.0, 2.5, 6.5, 4.5],
    "w": [2.0, 6.0, 1.5, 7.0, 3.0, 8.0, 6.5, 2.0, 8.5, 1.0, 5.5, 3.5],
    "u": [1.0, 4.0, 2.5, 5.0, 2.0, 3.0, 9.5, 6.0, 4.0, 3.5, 7.0, 8.0],
})

INFINITE_CALLS = {
    "estimation_plot": (lambda d: vc.estimation_plot(d, x="g", y="v", n_resamples=50), "v"),
    "benchmark_bar": (lambda d: vc.benchmark_bar(d, x="g", y="v"), "v"),
    "lollipop": (lambda d: vc.lollipop(d, x="g", y="v"), "v"),
    "lollipop sum": (lambda d: vc.lollipop(d, x="g", y="v", stat="sum"), "v"),
    # A group's median can be finite with inf in it; inf is rejected all the same.
    "lollipop median": (lambda d: vc.lollipop(d, x="g", y="v", stat="median"), "v"),
    "dumbbell": (lambda d: vc.dumbbell(d, label="k", start="v", end="w"), "w"),
    "divergent_bar": (lambda d: vc.divergent_bar(d, category="g", left="v", right="w"), "v"),
    "centered_bar": (lambda d: vc.centered_bar(d, x="g", y="v"), "v"),
    "waffle": (lambda d: vc.waffle(d, category="g", value="v"), "v"),
    "donut_grid": (lambda d: vc.donut_grid(d.head(3), columns=["v", "w"]), "w"),
    "nested_pie": (lambda d: vc.nested_pie(d, outer="g", inner="k", value="v"), "v"),
    "circular_bar": (lambda d: vc.circular_bar(d, label="k", value="v"), "v"),
    "waterfall": (lambda d: vc.waterfall(d, label="k", value="v"), "v"),
    "funnel": (lambda d: vc.funnel(d, stage="k", value="v"), "v"),
    "bullet target": (lambda d: vc.bullet(d.head(3), label="k", value="v", target="w"), "w"),
    "bullet bands": (lambda d: vc.bullet(d.head(3), label="k", value="v", bands=["u"]), "u"),
    "histogram": (lambda d: vc.histogram(d, x="v"), "v"),
    "ridgeplot": (lambda d: vc.ridgeplot(d, x="v", group="g"), "v"),
    "raincloud": (lambda d: vc.raincloud(d, x="g", y="v"), "v"),
    "correlogram": (lambda d: vc.correlogram(d, columns=["v", "w", "u"]), "u"),
    "correlogram default columns": (lambda d: vc.correlogram(d), "u"),
    # Spearman ranks are defined for inf; inf is rejected all the same.
    "correlogram spearman": (lambda d: vc.correlogram(d, columns=["v", "w", "u"], method="spearman"), "u"),
    "compare_correlations": (lambda d: vc.compare_correlations(d, group="g"), "w"),
    "quadrant_plot": (lambda d: vc.quadrant_plot(d, x="v", y="w"), "w"),
    "profile_boxes": (lambda d: vc.profile_boxes(d, categorical=["g"], numeric=["v", "w"]), "v"),
    "profile_scatters": (lambda d: vc.profile_scatters(d, columns=["v", "w"]), "w"),
    "animated_bubble": (lambda d: vc.animated_bubble(d, time="g", x="v", y="w", size="u"), "u"),
}


def _spoil(data, column, how):
    d = data.copy()
    if how == "all inf":
        d[column] = np.inf
        return d
    d.loc[1, column] = -np.inf if how == "one -inf" else np.inf  # row 1 is also in head(3)
    if how == "Float64":
        d[column] = d[column].astype("Float64")
    return d


@pytest.mark.parametrize("how", ["one inf", "one -inf", "all inf", "Float64"])
@pytest.mark.parametrize("name", list(INFINITE_CALLS))
def test_an_infinite_value_in_a_numeric_role_raises_naming_the_column(name, how):
    call, column = INFINITE_CALLS[name]
    call(NUMBERS)  # the clean data draws
    plt.close("all")
    with pytest.raises(ValueError, match=rf"infinite values \(inf or -inf\): \['{column}'\]"):
        call(_spoil(NUMBERS, column, how))
    assert plt.get_fignums() == []


def test_check_finite_allows_missing_values_and_types_that_cannot_be_infinite():
    d = pd.DataFrame({"f": [1.0, np.nan], "F": pd.array([1.5, None], dtype="Float64"),
                      "i": pd.array([1, None], dtype="Int64"), "b": [True, False], "n": [1, 2],
                      "c": [1 + 2j, np.nan], "big": [1e308, -1e308]})
    check_finite(d, *d.columns)
    with pytest.raises(ValueError, match=r"\['c'\]"):
        check_finite(d.assign(c=[complex(np.inf, 0), 1j]), "c")


def test_lollipop_counts_may_include_infinite_values():
    d = _spoil(NUMBERS, "v", "one inf")
    assert vc.lollipop(d, x="g", y="v", stat="count").table["value"].tolist() == [4, 4, 4]


def test_the_statistics_functions_keep_their_nan_results_for_infinite_input():
    assert np.isnan(st.correlation_test([np.inf] * 4, [1, 3, 2, 4])["r"])


# --- groups are selected from read-only object columns -------------------------------------------------------------


def _read_only(data):
    """*data* with its text columns held in one read-only object array, as a frame built with ``copy=False`` has.

    Categorical and string columns become object columns. Numbers and dates are copied.
    """
    text = [c for c in data.columns if not (pd.api.types.is_numeric_dtype(data[c])
                                            or pd.api.types.is_datetime64_any_dtype(data[c]))]
    labels = np.asfortranarray(data[text].to_numpy(dtype=object))  # each column a contiguous slice
    labels.flags.writeable = False
    out = pd.DataFrame(labels, columns=text, index=data.index, dtype=object, copy=False)
    for c in data.columns:
        if c not in text:
            out[c] = data[c].to_numpy(copy=True)
    return out


SURVEY_ITEMS = ["Workload is manageable", "My manager supports me", "Tools are adequate"]
READ_ONLY_CALLS = {
    "estimation_plot": ("trial", lambda d: vc.estimation_plot(d, x="arm", y="score", n_resamples=50)),
    "benchmark_bar": ("trial", lambda d: vc.benchmark_bar(d, x="arm", y="score")),
    "lollipop": ("trial", lambda d: vc.lollipop(d, x="arm", y="score")),
    "dumbbell": ("small", lambda d: vc.dumbbell(d, label="k", start="a", end="b")),
    "divergent_bar": ("small", lambda d: vc.divergent_bar(d, category="o", left="a", right="b")),
    "centered_bar": ("trial", lambda d: vc.centered_bar(d, x="arm", y="score")),
    "likert": ("survey", lambda d: vc.likert(d, items=SURVEY_ITEMS, levels=sorted(set(d[SURVEY_ITEMS[0]].dropna())))),
    "waffle": ("survey", lambda d: vc.waffle(d, category="team", value="tenure_years")),
    "percent_grid": ("trial", lambda d: vc.percent_grid(d, column="site", success="North", facet="arm")),
    "percent_grid without facet": ("trial", lambda d: vc.percent_grid(d, column="site", success="North")),
    "stacked_percentages": ("trial", lambda d: vc.stacked_percentages(d, group="arm", category="site")),
    "nested_pie": ("small", lambda d: vc.nested_pie(d, outer="o", inner="k", value="a")),
    "circular_bar": ("small", lambda d: vc.circular_bar(d, label="k", value="a", group="o")),
    "waterfall": ("small", lambda d: vc.waterfall(d, label="k", value="a")),
    "funnel": ("small", lambda d: vc.funnel(d, stage="k", value="a")),
    "bullet": ("small", lambda d: vc.bullet(d, label="k", value="a", target="b")),
    "histogram facet": ("trial", lambda d: vc.histogram(d, x="score", facet="arm")),
    "histogram hue": ("trial", lambda d: vc.histogram(d, x="score", hue="arm")),
    "ridgeplot": ("trial", lambda d: vc.ridgeplot(d, x="score", group="arm")),
    "raincloud": ("trial", lambda d: vc.raincloud(d, x="arm", y="score")),
    "compare_correlations": ("measurements", lambda d: vc.compare_correlations(d, group="group")),
    "quadrant_plot": ("trial", lambda d: vc.quadrant_plot(d, x="baseline", y="score", label="arm")),
    "profile_bars": ("survey", lambda d: vc.profile_bars(d)),
    "profile_boxes": ("trial", lambda d: vc.profile_boxes(d)),
    "pca_plot": ("measurements", lambda d: vc.pca_plot(d, features=["length", "width", "depth"], group="group")),
    "radar": ("measurements", lambda d: vc.radar(d, metrics=["length", "width", "depth"], group="group")),
    "gantt": ("projects", lambda d: vc.gantt(d, task="task", start="start", end="end", group="team")),
    "animated_bubble": ("trial", lambda d: vc.animated_bubble(d, time="site", x="baseline", y="score", size="baseline",
                                                              color="arm")),
}


def _frame(name):
    if name == "small":
        return pd.DataFrame({"k": list("abcd"), "o": list("xxyy"), "a": [3.0, 5, 2, 8], "b": [4.0, 6, 3, 9]})
    return getattr(datasets, name)()


@pytest.mark.parametrize("name", list(READ_ONLY_CALLS))
def test_charts_read_text_columns_backed_by_a_read_only_array(name):
    # pandas before 2.2 cannot compare a read-only object array with a scalar; copy-on-write passes such arrays on.
    source, call = READ_ONLY_CALLS[name]
    modes = [contextlib.nullcontext()]
    if int(pd.__version__.split(".")[0]) < 3:
        modes.append(pd.option_context("mode.copy_on_write", True))
    expected = call(_read_only(_frame(source)).copy()).table
    for mode in modes:
        with mode:
            d = _read_only(_frame(source))
            assert all(not d[c].to_numpy().flags.writeable for c in d.columns if d[c].dtype == object)
            res = call(d)
            if name == "animated_bubble":
                res.info["animation"]._func(1)  # draw a later frame too
            pd.testing.assert_frame_equal(res.table, expected)
        plt.close("all")


def test_level_mask_matches_rows_as_equality_does():
    dates = pd.Series([np.datetime64("2024-01-01"), np.datetime64("2024-01-02")], dtype=object)
    assert level_mask(dates, pd.Timestamp("2024-01-02")).tolist() == [False, True]  # isin([...]) finds neither
    flags = pd.Series([1, 0, None], dtype=object)
    assert level_mask(flags, True).tolist() == (flags == True).tolist() == [True, False, False]  # noqa: E712
    frozen = np.array(["a", "b", "a"], dtype=object)
    frozen.flags.writeable = False
    assert level_mask(pd.Series(frozen, copy=False), "a").tolist() == [True, False, True]


# --- animated_bubble skips bubbles it cannot draw ------------------------------------------------------------------


def _render_every_frame(res):
    for i in range(len(res.info["frames"])):
        res.info["animation"]._func(i)
        res.figure.savefig(BytesIO(), format="png")


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize("frame", [0, 2], ids=["first frame", "later frame"])
def test_animated_bubble_draws_a_group_whose_only_bubble_has_no_size(frame, dtype):
    d = pd.DataFrame({"t": [0, 0, 1, 1, 2, 2], "g": ["a", "b"] * 3, "x": [1.0, 2, 3, 4, 5, 6],
                      "y": [6.0, 5, 4, 3, 2, 1], "s": pd.array([1.0, 2, 3, 4, 5, 6], dtype=dtype)})
    d.loc[2 * frame, "s"] = None  # group a's only bubble in that frame
    res = vc.animated_bubble(d, time="t", x="x", y="y", size="s", color="g", label="g")
    _render_every_frame(res)
    res.info["animation"]._func(frame)
    texts = [t.get_text() for t in res.axes.texts]
    assert "a" not in texts and "b" in texts  # the size-less bubble is neither drawn nor labelled
    assert [t.get_text() for t in res.axes.get_legend().get_texts()] == ["a", "b"]


def test_animated_bubble_draws_a_frame_whose_only_row_has_no_size():
    d = pd.DataFrame({"t": [0, 1, 2], "x": [1.0, 2, 3], "y": [3.0, 2, 1], "s": [1.0, np.nan, 3]})
    res = vc.animated_bubble(d, time="t", x="x", y="y", size="s")
    _render_every_frame(res)
    res.info["animation"]._func(1)
    assert sum(len(c.get_offsets()) for c in res.axes.collections) == 0


# --- sums of nullable numbers come back in NumPy dtypes -------------------------------------------------------------


@pytest.mark.parametrize("values, dtype, expected", [
    ([1, 2, None, 4], "Int64", np.int64),
    ([1, 2, 3, 4], "Int64", np.int64),
    ([1.5, 2.0, None, 4.0], "Float64", np.float64),
    ([True, False, None, True], "boolean", np.int64),
], ids=["Int64 with NA", "Int64 complete", "Float64 with NA", "boolean with NA"])
def test_waffle_and_nested_pie_tables_hold_numpy_numbers(values, dtype, expected):
    d = pd.DataFrame({"c": list("abab"), "o": list("xxyy"), "v": pd.array(values, dtype=dtype)})
    waffle = vc.waffle(d, category="c", value="v", order=["a", "b"]).table
    nested = vc.nested_pie(d, outer="o", inner="c", value="v").table
    for table in (waffle, nested):
        assert table["value"].dtype == expected
        assert "value" in table.select_dtypes("number").columns
    total = np.nansum(np.array(values, dtype=float))
    assert waffle["value"].sum() == nested["value"].sum() == total


def test_safe_sum_groups_by_several_keys_and_keeps_missing_keys():
    v = pd.Series([1, 2, None, 4], dtype="Int64")
    a, b = pd.Series(["x", "x", None, "y"]), pd.Series(["p", "q", "p", None])
    kept = safe_sum(v, [a, b], dropna=False)
    assert kept.dtype == np.int64 and kept.sum() == 7 and len(kept) == 4
    dropped = safe_sum(v, [a, b])
    assert dropped.dtype == np.int64 and dropped.to_dict() == {("x", "p"): 1, ("x", "q"): 2}


# --- a label that selects more than one column ---------------------------------------------------------------------

LIKERT_LEVELS = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
REPEATED_CALLS = {
    "waffle": ("survey", "team", lambda d: vc.waffle(d, category="team")),
    "percent_grid": ("trial", "arm", lambda d: vc.percent_grid(d, column="improved", facet="arm")),
    "stacked_percentages": ("trial", "site", lambda d: vc.stacked_percentages(d, group="arm", category="site")),
    "donut_grid default columns": ("small", "a", lambda d: vc.donut_grid(d)),
    "nested_pie": ("small", "a", lambda d: vc.nested_pie(d, outer="o", inner="k", value="a")),
    "circular_bar": ("small", "k", lambda d: vc.circular_bar(d, label="k", value="a")),
    "waterfall": ("small", "a", lambda d: vc.waterfall(d, label="k", value="a")),
    "funnel": ("small", "k", lambda d: vc.funnel(d, stage="k", value="a")),
    "bullet": ("small", "b", lambda d: vc.bullet(d, label="k", value="a", target="b")),
    "estimation_plot": ("trial", "score", lambda d: vc.estimation_plot(d, x="arm", y="score", n_resamples=50)),
    "benchmark_bar": ("trial", "arm", lambda d: vc.benchmark_bar(d, x="arm", y="score", threshold=50)),
    "lollipop": ("trial", "arm", lambda d: vc.lollipop(d, x="arm", y="score")),
    "dumbbell": ("small", "k", lambda d: vc.dumbbell(d, label="k", start="a", end="b")),
    "divergent_bar": ("small", "b", lambda d: vc.divergent_bar(d, category="k", left="a", right="b")),
    "centered_bar": ("trial", "score", lambda d: vc.centered_bar(d, x="arm", y="score")),
    "likert": ("survey", "Tools are adequate",
               lambda d: vc.likert(d, items=["Tools are adequate"], levels=LIKERT_LEVELS)),
    "correlogram default columns": ("measurements", "length", lambda d: vc.correlogram(d)),
    "compare_correlations default columns": ("measurements", "length",
                                             lambda d: vc.compare_correlations(d, group="group")),
    "quadrant_plot": ("trial", "score", lambda d: vc.quadrant_plot(d, x="baseline", y="score")),
    "histogram": ("trial", "score", lambda d: vc.histogram(d, x="score")),
    "ridgeplot": ("trial", "arm", lambda d: vc.ridgeplot(d, x="score", group="arm")),
    "raincloud": ("trial", "score", lambda d: vc.raincloud(d, x="arm", y="score")),
    "profile_bars default columns": ("survey", "team", lambda d: vc.profile_bars(d)),
    "profile_boxes default columns": ("trial", "arm", lambda d: vc.profile_boxes(d)),
    "profile_scatters default columns": ("measurements", "length", lambda d: vc.profile_scatters(d)),
    "pca": ("measurements", "width", lambda d: vc.pca(d, features=["length", "width", "depth"])),
    "pca_plot": ("measurements", "group", lambda d: vc.pca_plot(d, features=["length", "width"], group="group")),
    "radar": ("measurements", "group", lambda d: vc.radar(d, metrics=["length", "width", "depth"], group="group")),
    "sankey": ("network", "strength", lambda d: vc.sankey(d, source="source", target="target", value="strength")),
    "period_bars": ("sales", "online", lambda d: vc.period_bars(d, date="date", value="online")),
    "timeseries_fill": ("sales", "date", lambda d: vc.timeseries_fill(d, time="date", series=["online", "store"])),
    "calendar_heatmap": ("sales", "date", lambda d: vc.calendar_heatmap(d, date="date", value="online")),
    "gantt": ("projects", "start", lambda d: vc.gantt(d, task="task", start="start", end="end")),
    "duration_plot": ("projects", "end", lambda d: vc.duration_plot(d, label="task", start="start", end="end",
                                                                    inner_start="work_start", inner_end="work_end")),
    "animated_bubble": ("trial", "score", lambda d: vc.animated_bubble(d, time="site", x="baseline", y="score",
                                                                      size="baseline")),
}


def _repeat(data, column):
    return pd.concat([data, data[[column]]], axis=1)


@pytest.mark.parametrize("name", list(REPEATED_CALLS))
def test_a_repeated_label_in_a_role_raises_naming_it(name):
    source, column, call = REPEATED_CALLS[name]
    if name == "sankey":
        pytest.importorskip("plotly")
    with pytest.raises(ValueError, match=rf"column '{column}' appears 2 times in data; give each column its own label"):
        call(_repeat(_frame(source), column))
    assert plt.get_fignums() == []


def test_network_map_rejects_a_repeated_label():
    pytest.importorskip("networkx")
    d = _repeat(datasets.network(), "strength")
    with pytest.raises(ValueError, match="column 'strength' appears 2 times"):
        vc.network_map(d, source="source", target="target", weight="strength")


@pytest.mark.parametrize("call", [
    lambda d: vc.benchmark_bar(d, x="arm", y="score"),
    lambda d: vc.profile_boxes(d, categorical=["arm"], numeric=["score"]),  # profile_boxes scans every column
    lambda d: vc.correlogram(d, columns=["baseline", "score"]),
], ids=["benchmark_bar", "profile_boxes", "correlogram"])
def test_a_repeated_label_the_chart_does_not_use_is_ignored(trial, call):
    expected = call(trial).table
    pd.testing.assert_frame_equal(call(_repeat(trial, "site")).table, expected)


def test_a_role_label_with_a_zero_false_twin_raises_naming_both(trial):
    d = pd.DataFrame({"arm": trial["arm"], "score": trial["score"], "flat": 1000.0})
    d.columns = pd.Index(["arm", 0, False], dtype=object)
    for y in (0, False):
        with pytest.raises(ValueError, match=r"column \w+ matches 2 columns in data, \[0, False\]: pandas treats"):
            vc.estimation_plot(d, x="arm", y=y)  # read column False's values for y=0 before
    assert vc.estimation_plot(d.iloc[:, :2], x="arm", y=0).table["mean"].tolist() == pytest.approx(
        trial.groupby("arm", observed=True)["score"].mean().tolist())


@pytest.mark.parametrize("call, name", [
    (lambda d: vc.profile_boxes(d, categorical=[None], numeric=["width"]), "categorical"),
    (lambda d: vc.correlogram(d, columns=["length", None]), "columns"),
    (lambda d: vc.pca(d, features=["length", "width", None]), "features"),
    (lambda d: vc.radar(d, metrics=[None, "width"], group="group"), "metrics"),
], ids=["profile_boxes", "correlogram", "pca", "radar"])
def test_none_in_a_list_of_columns_is_rejected(call, name):
    with pytest.raises(ValueError, match=f"{name} contains None, which names no column"):
        call(datasets.measurements())


class _Frame:
    """Stands in for a polars DataFrame or a pyarrow Table: not pandas, but convertible with ``to_pandas``."""

    def to_pandas(self):
        return datasets.trial()


def test_a_frame_from_another_library_gets_a_conversion_hint():
    with pytest.raises(TypeError, match=r"got test_input_validation\._Frame; convert it first with \.to_pandas\(\)"):
        vc.lollipop(_Frame(), x="arm", y="score")
    with pytest.raises(TypeError, match=r"data must be a pandas DataFrame, got list$"):
        vc.lollipop([1, 2], x="arm", y="score")
    pa = pytest.importorskip("pyarrow")
    with pytest.raises(TypeError, match=r"got pyarrow\.lib\.Table; convert it first with \.to_pandas\(\)"):
        vc.lollipop(pa.table({"arm": ["a"], "score": [1.0]}), x="arm", y="score")


# --- complex numbers ------------------------------------------------------------------------------------------------

# These calls pick their columns by default, and a complex column is not picked (see the next tests).
COMPLEX_CALLS = [name for name in INFINITE_CALLS if name not in ("correlogram default columns", "compare_correlations")]


@pytest.mark.parametrize("name", COMPLEX_CALLS)
def test_a_complex_column_in_a_numeric_role_raises_naming_it(name):
    call, column = INFINITE_CALLS[name]
    d = NUMBERS.assign(**{column: NUMBERS[column] + 1j})
    with pytest.raises(TypeError, match=rf"column\(s\) must hold real numbers, not complex: \['{column}'\]"):
        call(d)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("call", [
    lambda d: vc.pca(d, ["v", "w", "u"]),
    lambda d: vc.period_bars(d.assign(t=pd.date_range("2024-01-01", periods=12)), date="t", value="u"),
], ids=["pca", "period_bars"])
def test_more_charts_reject_complex_columns(call):
    with pytest.raises(TypeError, match=r"not complex: \['u'\]"):
        call(NUMBERS.assign(u=NUMBERS["u"] * 1j))


def test_charts_leave_complex_columns_out_of_their_default_columns():
    d = NUMBERS.assign(z=NUMBERS["u"] + 1j)
    assert list(vc.correlogram(d).info["matrix"].columns) == ["v", "w", "u"]
    assert vc.donut_grid(d.head(3)).table["category"].unique().tolist() == ["v", "w", "u"]
    assert set(vc.profile_scatters(d, min_levels=2).table[["x", "y"]].stack()) == {"v", "w", "u"}
    assert set(vc.compare_correlations(d, group="g").table[["var1", "var2"]].stack()) == {"v", "w", "u"}


@pytest.mark.parametrize("call, name", [
    (lambda v: st.mean_ci(v), "x"),
    (lambda v: st.welch_test([1.0, 2.0, 3.0], v), "b"),
    (lambda v: st.hedges_g(v, [1.0, 2.0, 3.0]), "a"),
    (lambda v: st.bootstrap_ci(v, n_resamples=10), "samples"),
    (lambda v: st.correlation_test([1.0, 2.0, 3.0], v), "y"),
    (lambda v: st.histogram_bins(v), "x"),
    (lambda v: st.largest_remainder(v, 10), "values"),
], ids=["mean_ci", "welch_test", "hedges_g", "bootstrap_ci", "correlation_test", "histogram_bins", "largest_remainder"])
@pytest.mark.parametrize("kind", ["array", "Series", "list"])
def test_statistics_reject_complex_numbers(call, name, kind):
    values = [1 + 2j, 3 + 0j, 4 + 1j]
    values = {"array": np.array(values), "Series": pd.Series(values), "list": values}[kind]
    with pytest.raises(TypeError, match=f"{name} must be real numbers, not complex"):
        call(values)


# --- inputs to the statistics functions -----------------------------------------------------------------------------


@pytest.mark.parametrize("call, message", [
    (lambda: st.mean_ci("123"), "x must be numbers, not the str '123'"),
    (lambda: st.mean_ci(b"5"), "x must be numbers, not the bytes b'5'"),
    (lambda: st.welch_test("12", "34"), "a must be numbers"),
    (lambda: st.adjust_pvalues("0.5"), "p must be numbers"),
    (lambda: st.histogram_bins("5"), "x must be numbers"),
    (lambda: st.largest_remainder("12", 3), "values must be numbers"),
    (lambda: st.wilson_ci("12", 40), "successes must be numbers"),
    (lambda: st.correlation_test([1.0, 2.0, 3.0], "123"), "y must be numbers"),
], ids=["mean_ci", "mean_ci bytes", "welch_test", "adjust_pvalues", "histogram_bins", "largest_remainder",
        "wilson_ci", "correlation_test"])
def test_statistics_do_not_read_a_string_as_a_number(call, message):
    with pytest.raises(TypeError, match=message):
        call()


def test_statistics_read_iterables_that_are_not_sequences():
    expected = st.mean_ci([1.0, 2.0, 4.0])
    assert st.mean_ci(v for v in [1.0, 2.0, 4.0]) == expected
    assert st.mean_ci({"a": 1.0, "b": 2.0, "c": 4.0}.values()) == expected
    assert st.mean_ci(map(float, [1, 2, 4])) == expected
    pairs = st.correlation_test((v for v in [1.0, 2.0, 3.0, 4.0]), iter([1.0, 2.0, 3.0, 5.0]))
    assert pairs == st.correlation_test([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0])


@pytest.mark.parametrize("x, y", [
    ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]),
    ([1.0, 2.0, 3.0, 4.0], [1.0]),  # raised an IndexError
], ids=["3 and 4", "4 and 1"])
def test_correlation_test_needs_paired_values(x, y):
    with pytest.raises(ValueError, match=r"x and y must have the same length, since their values are paired"):
        st.correlation_test(x, y)


def test_correlation_test_rejects_unordered_values():
    with pytest.raises(TypeError, match="x must be a sequence such as a list or an array, not a set"):
        st.correlation_test({1.0, 2.0, 3.0, 4.0}, [1.0, 2.0, 3.0, 5.0])
    with pytest.raises(TypeError, match="y must be a sequence .* not a dict_values"):
        st.correlation_test([1.0, 2.0, 3.0, 4.0], {"a": 1.0, "b": 2.0, "c": 3.0, "d": 5.0}.values())


def test_elementwise_statistics_name_their_arguments_when_shapes_differ():
    with pytest.raises(ValueError, match=r"successes and n must have the same length.*\(3,\) and \(2,\)"):
        st.wilson_ci([1, 2, 3], [10, 10])
    with pytest.raises(ValueError, match=r"r1, n1, r2 and n2 must have matching shapes.*r1 \(3,\), n1 \(2,\)"):
        st.compare_correlations_test([0.1, 0.2, 0.3], [10, 10], [0.3, 0.4], 10)
    low, high = st.wilson_ci([1, 2, 3], 10)  # a single number still pairs with every element
    assert low.shape == high.shape == (3,)
    z, p = st.compare_correlations_test([0.1, 0.2], 10, [0.3, 0.4], 10)
    assert z.shape == p.shape == (2,)


# --- colour, colormap and count arguments ---------------------------------------------------------------------------

SINGLE_COLOUR_CALLS = {
    "lollipop": lambda ax, **kw: vc.lollipop(datasets.survey(), x="team", y="tenure_years", ax=ax, **kw),
    "funnel": lambda ax, **kw: vc.funnel(SMALL, stage="k", value="a", ax=ax, **kw),
    "bullet": lambda ax, **kw: vc.bullet(SMALL.head(1), label="k", value="a", ax=ax, **kw),
    "upset": lambda ax, **kw: vc.upset(datasets.memberships(), **kw),
    "quadrant_plot": lambda ax, **kw: vc.quadrant_plot(datasets.trial(), x="baseline", y="score", ax=ax, **kw),
    "profile_bars": lambda ax, **kw: vc.profile_bars(datasets.survey(), columns=["team"], **kw),
    "period_bars": lambda ax, **kw: vc.period_bars(datasets.sales().head(60), date="date", value="online", ax=ax,
                                                   **kw),
}


@pytest.mark.parametrize("value", ["x", ["red", "blue"], ["red"]], ids=["unknown", "two colours", "a list of one"])
@pytest.mark.parametrize("name", list(SINGLE_COLOUR_CALLS))
def test_a_single_colour_argument_must_be_one_colour(name, value):
    argument = "bar_color" if name == "bullet" else "color"
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match=rf"{argument} must be one colour, such as 'red'.*got {re.escape(repr(value))}"):
        SINGLE_COLOUR_CALLS[name](ax, **{argument: value})
    assert not (ax.lines or ax.collections or ax.patches)  # nothing drawn on the caller's Axes
    assert plt.get_fignums() == [fig.number]
    plt.close("all")
    SINGLE_COLOUR_CALLS[name](None, **{argument: (0.2, 0.4, 0.6)})  # an RGB tuple is one colour


@pytest.mark.parametrize("call, argument", [
    (lambda **kw: vc.correlogram(datasets.measurements(), **kw), "cmap"),
    (lambda **kw: vc.compare_correlations(datasets.measurements(), group="group", **kw), "cmap"),
    (lambda **kw: vc.calendar_heatmap(datasets.sales().head(60), date="date", value="online", **kw), "cmap"),
], ids=["correlogram", "compare_correlations", "calendar_heatmap"])
def test_a_colormap_argument_must_name_a_colormap(call, argument):
    with pytest.raises(ValueError, match=r"cmap must be a Matplotlib colormap name such as 'viridis', got 'C0', "
                                         "which is a colour, not a colormap"):
        call(**{argument: "C0"})
    with pytest.raises(ValueError, match=r"cmap must be a Matplotlib colormap name such as 'viridis', got 'nope'$"):
        call(**{argument: "nope"})
    assert plt.get_fignums() == []
    call(**{argument: plt.get_cmap("viridis")})  # a Colormap object is still accepted


def test_calendar_heatmap_checks_its_missing_colour():
    with pytest.raises(ValueError, match="missing_color must be one colour, such as 'red'.*got 'nope'"):
        vc.calendar_heatmap(datasets.sales().head(60), date="date", value="online", missing_color="nope")


@pytest.mark.parametrize("call", [
    lambda: vc.palette(3, "notacmap"),
    lambda: vc.waffle(datasets.survey(), category="team", colors="notacmap"),
    lambda: vc.ridgeplot(datasets.measurements(), x="depth", group="group", colors="notacmap"),
], ids=["palette", "waffle", "ridgeplot"])
def test_colors_name_the_argument_for_an_unknown_colormap(call):
    with pytest.raises(ValueError, match="colors must be a Matplotlib colormap name or a list of colours, "
                                         "got 'notacmap'$"):
        call()


def test_palette_suggests_a_list_for_one_colour():
    with pytest.raises(ValueError, match=r"got 'red'; for one colour use \['red'\]"):
        vc.palette(3, "red")
    assert vc.palette(3, ["red"]) == ["red"] * 3


@pytest.mark.parametrize("n", [2.5, 2.0, -2, True, "3"])
def test_palette_needs_a_whole_number_of_colours(n):
    with pytest.raises(ValueError, match="n must be a whole number of at least 0"):
        vc.palette(n)


def test_palette_takes_zero_and_numpy_integers():
    assert vc.palette(0) == []
    assert vc.palette(np.int64(2)) == list(vc.OKABE_ITO[:2])


@pytest.mark.parametrize("call, name", [
    (lambda: vc.waffle(datasets.survey(), category="team", rows=2.0), "rows"),
    (lambda: vc.waffle(datasets.survey(), category="team", columns=5.0), "columns"),
    (lambda: vc.percent_grid(datasets.trial(), column="improved", facet="arm", col_wrap=2.0), "col_wrap"),
    (lambda: vc.donut_grid(SMALL, col_wrap=2.0), "col_wrap"),
    (lambda: vc.circular_bar(SMALL, label="k", value="a", gap=1.0), "gap"),
    (lambda: vc.upset(datasets.memberships(), max_intersections=3.0), "max_intersections"),
    (lambda: vc.histogram(datasets.trial(), x="score", facet="arm", col_wrap=2.0), "col_wrap"),
], ids=["waffle rows", "waffle columns", "percent_grid", "donut_grid", "circular_bar", "upset", "histogram"])
def test_counts_given_as_whole_floats_get_the_same_message_everywhere(call, name):
    with pytest.raises(ValueError, match=rf"{name} must be a whole number of at least \d given as an int, "
                                         rf"got the float \d\.0; use \d$"):
        call()


# --- no row holds every value a chart needs -------------------------------------------------------------------------


def _apart(data, first, second):
    """*data* with *first* missing in its first half and *second* in its second half: no row has both."""
    d = data.copy().reset_index(drop=True)
    half = len(d) // 2
    d.loc[: half - 1, first] = None
    d.loc[half:, second] = None
    return d


@pytest.mark.parametrize("call, message", [
    (lambda: vc.correlogram(_apart(datasets.measurements(), "length", "width"), columns=["length", "width"]),
     r"no row has a value in two of the columns \['length', 'width'\]"),
    (lambda: vc.compare_correlations(_apart(datasets.measurements(), "length", "width"), group="group",
                                     columns=["length", "width"]),
     r"no row has a value in two of the columns \['length', 'width'\]"),
    (lambda: vc.gantt(_apart(datasets.projects(), "start", "end"), task="task", start="start", end="end"),
     r"no row has a value in each of \['start', 'end'\]"),
    (lambda: vc.duration_plot(_apart(datasets.projects(), "start", "end"), "task", "start", "end", "work_start",
                              "work_end"),
     r"no row has a value in each of \['start', 'end'\]"),
    (lambda: vc.raincloud(_apart(datasets.trial(), "arm", "score"), x="arm", y="score"),
     r"no row has a value in each of \['arm', 'score'\]"),
    (lambda: vc.timeseries_fill(_apart(datasets.sales(), "online", "store"), time="date", series=["online", "store"]),
     "no time has both a 'online' and a 'store' value"),
], ids=["correlogram", "compare_correlations", "gantt", "duration_plot", "raincloud", "timeseries_fill"])
def test_a_chart_with_no_complete_row_raises_before_drawing(call, message):
    with pytest.raises(ValueError, match=message):
        call()
    assert plt.get_fignums() == []


def test_charts_still_draw_when_some_rows_are_complete():
    d = _apart(datasets.measurements(), "length", "width")
    d.loc[0, "length"] = 5.0
    d.loc[1, "length"] = 4.0  # two complete rows
    assert vc.correlogram(d, columns=["length", "width", "depth"]).table["n"].tolist() == [2, 77, 75]
    p = datasets.projects()
    p.loc[:2, "end"] = pd.NaT
    assert vc.gantt(p, task="task", start="start", end="end").table["duration_days"].notna().sum() == 3
    inner = datasets.projects().assign(work_start=pd.NaT)  # a missing inner window is documented: no inner bar
    assert vc.duration_plot(inner, "task", "start", "end", "work_start", "work_end").table["inner_days"].isna().all()
    # Rows of one time are averaged before the series are compared, so values on different rows still pair up.
    long = pd.DataFrame({"t": [1, 1, 2, 2], "a": [1.0, None, 3.0, None], "b": [None, 2.0, None, 1.0]})
    assert vc.timeseries_fill(long, time="t", series=["a", "b"]).table["difference"].tolist() == [-1.0, 2.0]


def test_profile_scatters_marks_a_pair_with_no_complete_rows():
    d = _apart(datasets.measurements(), "length", "width")
    res = vc.profile_scatters(d, columns=["length", "width", "depth"])
    empty = res.table.index[res.table["n"] == 0].tolist()
    assert len(empty) == 1
    ax = res.figure[0].axes[empty[0]]
    assert ax.get_title(loc="left") == "n=0"
    assert [t.get_text() for t in ax.texts] == ["no complete rows"]


# --- a role column with no values -----------------------------------------------------------------------------------

MEASURES = ["length", "width", "depth", "mass"]


@pytest.mark.parametrize("call, column", [
    (lambda: vc.dumbbell(SMALL.assign(k=np.nan), label="k", start="a", end="b"), "k"),
    (lambda: vc.gantt(datasets.projects().assign(task=np.nan), task="task", start="start", end="end"), "task"),
    (lambda: vc.duration_plot(datasets.projects().assign(task=np.nan), "task", "start", "end", "work_start",
                              "work_end"), "task"),
    (lambda: vc.quadrant_plot(datasets.measurements().assign(lab=np.nan), x="length", y="width", label="lab"), "lab"),
    (lambda: vc.radar(datasets.measurements().assign(length=np.nan), metrics=MEASURES, group="group"), "length"),
    (lambda: vc.pca(datasets.measurements().assign(length=np.nan), MEASURES), "length"),
    (lambda: vc.pca_plot(datasets.measurements().assign(length=np.nan), MEASURES, group="group"), "length"),
    (lambda: vc.profile_scatters(datasets.measurements().assign(length=np.nan), columns=["length", "width"]),
     "length"),
], ids=["dumbbell", "gantt", "duration_plot", "quadrant_plot", "radar", "pca", "pca_plot", "profile_scatters"])
def test_an_all_missing_role_column_raises_naming_it(call, column):
    with pytest.raises(ValueError, match=f"column '{column}' has no non-missing values"):
        call()
    assert plt.get_fignums() == []


def test_quadrant_plot_writes_no_label_for_a_point_without_one():
    d = datasets.measurements().head(10).assign(lab=list("abcdefghi") + [None])
    texts = [t.get_text() for t in vc.quadrant_plot(d, x="length", y="width", label="lab").axes.texts]
    assert "nan" not in texts and "None" not in texts
    assert sum(len(t) == 1 for t in texts) == 9


# --- options -------------------------------------------------------------------------------------------------------

SORT_CALLS = {
    "benchmark_bar": lambda sort: vc.benchmark_bar(datasets.trial(), x="arm", y="score", sort=sort),
    "dumbbell": lambda sort: vc.dumbbell(SMALL, label="k", start="a", end="b", sort=sort),
    "divergent_bar": lambda sort: vc.divergent_bar(SMALL, category="k", left="a", right="b", sort=sort),
    "likert": lambda sort: vc.likert(datasets.survey(), items=["Tools are adequate"], levels=LIKERT_LEVELS, sort=sort),
    "circular_bar": lambda sort: vc.circular_bar(SMALL, label="k", value="a", sort=sort),
    "gantt": lambda sort: vc.gantt(datasets.projects(), task="task", start="start", end="end", sort=sort),
    "duration_plot": lambda sort: vc.duration_plot(datasets.projects(), "task", "start", "end", "work_start",
                                                   "work_end", sort=sort),
}


@pytest.mark.parametrize("sort", ["asc", "no", "False", None])
@pytest.mark.parametrize("name", list(SORT_CALLS))
def test_a_sort_flag_must_be_true_or_false(name, sort):
    with pytest.raises(ValueError, match=rf"sort must be one of \[True, False\], got {re.escape(repr(sort))}"):
        SORT_CALLS[name](sort)
    assert plt.get_fignums() == []
    SORT_CALLS[name](np.False_)


@pytest.mark.parametrize("bw_method", ["asc", 0.0, -1.0, np.nan, np.inf, True, [0.5]])
@pytest.mark.parametrize("call", [
    lambda bw: vc.ridgeplot(datasets.measurements(), x="depth", group="group", bw_method=bw),
    lambda bw: vc.raincloud(datasets.trial(), x="arm", y="score", bw_method=bw),
], ids=["ridgeplot", "raincloud"])
def test_bw_method_is_checked_before_drawing(call, bw_method):
    with pytest.raises(ValueError, match="bw_method must be 'scott', 'silverman', a positive number or a callable"):
        call(bw_method)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("bw_method", [None, "scott", "silverman", 0.5, Fraction(1, 2), lambda kde: 0.4])
def test_bw_method_accepts_what_scipy_accepts(bw_method):
    vc.ridgeplot(datasets.measurements(), x="depth", group="group", bw_method=bw_method)


@pytest.mark.parametrize("threshold", ["50", np.nan, np.inf, True])
def test_lollipop_threshold_must_be_a_finite_number(threshold):
    with pytest.raises(ValueError, match="threshold must be a finite number or None"):
        vc.lollipop(datasets.survey(), x="team", y="tenure_years", threshold=threshold)


@pytest.mark.parametrize("ref_line", [np.nan, np.inf, -np.inf])
def test_histogram_ref_line_must_be_finite(ref_line):
    with pytest.raises(ValueError, match="ref_line must be 'mean', 'median', a number or None.*must be finite"):
        vc.histogram(datasets.trial(), x="score", ref_line=ref_line)


@pytest.mark.parametrize("value", [Fraction(1, 2), Decimal("0.5"), np.float32(0.5)], ids=["Fraction", "Decimal",
                                                                                          "float32"])
def test_exact_numbers_are_accepted_by_every_real_option(value):
    expected = vc.circular_bar(SMALL, label="k", value="a", inner_radius=0.5).axes.get_ylim()
    assert vc.circular_bar(SMALL, label="k", value="a", inner_radius=value).axes.get_ylim() == expected
    vc.donut_grid(SMALL, min_label=value)
    vc.stacked_percentages(datasets.trial(), group="arm", category="site", min_label=value)
    vc.timeseries_fill(datasets.sales().head(30), time="date", series=["online", "store"], alpha=value)


def test_stacked_percentages_rejects_an_order_given_twice():
    with pytest.raises(TypeError, match="give category_order only; category_order_ is its deprecated alias"):
        vc.stacked_percentages(datasets.survey(), group="team", category="remote", category_order=[False, True],
                               category_order_=[True, False])
