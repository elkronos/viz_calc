"""Input checks shared by the charts: one column for two roles, infinite values and sums of nullable numbers."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets
from viz_calc import stats as st
from viz_calc._core import check_finite, safe_sum

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


# --- sums of nullable numbers come back in NumPy dtypes -------------------------------------------------------------


@pytest.mark.parametrize("values, dtype, expected", [
    ([1, 2, None, 4], "Int64", np.int64),
    ([1, 2, 3, 4], "Int64", np.int64),
    ([1.5, 2.0, None, 4.0], "Float64", np.float64),
], ids=["Int64 with NA", "Int64 complete", "Float64 with NA"])
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
