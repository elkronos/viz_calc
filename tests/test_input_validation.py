"""Input checks shared by the charts: one column for two roles and sums of nullable numbers."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets
from viz_calc._core import safe_sum

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
