"""Repeated Likert levels, significance and layout options, threshold labels, wrap-free sums and boolean labels."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Annotation

import viz_calc as vc
from viz_calc import datasets as ds
from viz_calc import stats as st
from viz_calc._core import check_range, safe_sum


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _numbers(n=30, seed=0):
    return pd.DataFrame(np.random.default_rng(seed).normal(size=(n, 3)), columns=list("abc"))


# --- likert ---------------------------------------------------------------------------------------------------------


def test_likert_rejects_a_repeated_level():
    d = pd.DataFrame({"q": ["no", "yes", "yes", "meh"]})
    assert vc.likert(d, items=["q"], levels=["no", "meh", "yes"]).table["net"].tolist() == [25.0]
    with pytest.raises(ValueError, match="repeats response option.*'yes'"):
        vc.likert(d, items=["q"], levels=["no", "meh", "yes", "yes"])


# --- significance level and other numeric options -------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [5, 1, 0, -0.1, 1.5, "0.05", True, np.nan, None])
def test_correlation_alpha_must_be_a_proportion(alpha):
    df = _numbers()
    with pytest.raises(ValueError, match="alpha must be strictly between 0 and 1"):
        vc.correlogram(df, alpha=alpha)
    with pytest.raises(ValueError, match="alpha must be strictly between 0 and 1"):
        vc.compare_correlations(df.assign(g=["x"] * 15 + ["y"] * 15), "g", alpha=alpha)


def test_correlation_alpha_in_range_still_works():
    df = _numbers()
    assert not vc.correlogram(df, alpha=0.01).table["significant"].any()


@pytest.mark.parametrize("decimals", [-1, 2.0, "2", True])
def test_correlation_decimals_must_be_a_whole_number(decimals):
    df = _numbers()
    with pytest.raises(ValueError, match="decimals must be a whole number"):
        vc.correlogram(df, decimals=decimals)
    with pytest.raises(ValueError, match="decimals must be a whole number"):
        vc.compare_correlations(df.assign(g=["x"] * 15 + ["y"] * 15), "g", decimals=decimals)


def test_correlation_decimals_zero_is_allowed():
    r = vc.correlogram(_numbers(), decimals=0)
    assert all("." not in t.get_text() for t in r.axes.texts)


@pytest.mark.parametrize("components", [(1.0, 2.0), (1,), (1, 2, 3), 1, ("1", "2"), (True, 2), (0, 1)])
def test_pca_components_must_be_a_pair_of_component_numbers(components):
    m = ds.measurements()
    with pytest.raises(ValueError, match="components"):
        vc.pca_plot(m, ["length", "width", "depth", "mass"], components=components)


@pytest.mark.parametrize("loadings", [-1, 1.5, "2", None])
def test_pca_loadings_must_be_a_count_or_bool(loadings):
    m = ds.measurements()
    with pytest.raises(ValueError, match="loadings must be a whole number"):
        vc.pca_plot(m, ["length", "width", "depth", "mass"], loadings=loadings)


@pytest.mark.parametrize("loadings, arrows", [(0, 0), (2, 2), (True, 4), (np.True_, 4), (False, 0)])
def test_pca_loadings_draw_that_many_arrows(loadings, arrows):
    m = ds.measurements()
    r = vc.pca_plot(m, ["length", "width", "depth", "mass"], loadings=loadings, ellipse=None)
    assert sum(isinstance(t, Annotation) for t in r.axes.texts) == arrows


def test_other_numeric_options_are_checked():
    trial = ds.trial()
    with pytest.raises(ValueError, match="n_resamples"):
        vc.estimation_plot(trial, "arm", "score", n_resamples=0)
    with pytest.raises(ValueError, match="overlap must be a number in"):
        vc.ridgeplot(trial, "score", "arm", overlap=1)
    with pytest.raises(ValueError, match="alpha must be a number in"):
        vc.histogram(trial, "score", alpha=1.5)
    with pytest.raises(ValueError, match="max_levels"):
        vc.profile_bars(trial, max_levels=0)
    with pytest.raises(ValueError, match="max_levels"):
        vc.profile_boxes(trial, max_levels=2.5)
    with pytest.raises(ValueError, match="min_levels"):
        vc.profile_scatters(trial, min_levels=-1)
    with pytest.raises(ValueError, match="dpi"):
        vc.to_pptx([], "unused.pptx", dpi=0)


def test_check_range():
    assert check_range("x", 0.5, 0, 1) == 0.5
    assert check_range("x", 1, 0, 1) == 1.0
    for bad in (1, "0.5", True, np.nan, [0.5]):
        with pytest.raises(ValueError, match=r"x must be a number in \[0, 1\)"):
            check_range("x", bad, 0, 1, include_high=False)


# --- threshold labels -----------------------------------------------------------------------------------------------


def test_centered_bar_label_matches_the_classification():
    df = pd.DataFrame({"g": ["a"] * 3, "y": [0.9996, 0.99955, 0.9994]})
    r = vc.centered_bar(df, "g", "y", threshold=0.99955)
    assert [t.get_text() for t in r.axes.get_legend().get_texts()] == ["≥ 0.99955", "< 0.99955"]
    assert r.table["n_above"].tolist() == [2]


def test_threshold_labels_are_not_in_scientific_notation():
    df = pd.DataFrame({"store": ["A"] * 4, "sales": [1231, 1232, 1236, 1240]})
    r = vc.centered_bar(df, "store", "sales", threshold=1234.5)
    assert "1,234.5" in r.axes.get_title(loc="left")
    df = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4, "y": [1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238]})
    assert "vs benchmark 1,234.5 " in vc.benchmark_bar(df, "g", "y", threshold=1234.5).axes.get_title(loc="left")
    assert "vs benchmark 1,500 " in vc.benchmark_bar(df, "g", "y", threshold=1500).axes.get_title(loc="left")


def test_a_long_mean_is_rounded_only_as_far_as_the_data_allow():
    df = pd.DataFrame({"g": ["a"] * 3, "y": [1.0, 2.0, 2.0]})  # mean 5/3
    assert vc.centered_bar(df, "g", "y").axes.get_legend().get_texts()[0].get_text() == "≥ 1.66667"
    # 1.0000001 is below the mean 1.00000013..., but above its 6-digit rounding 1: more digits are needed.
    df = pd.DataFrame({"g": ["a"] * 3, "y": [1.0, 1.0000001, 1.0000003]})
    r = vc.centered_bar(df, "g", "y")
    shown = float(r.axes.get_legend().get_texts()[0].get_text()[2:].replace(",", ""))
    assert r.table["n_above"].tolist() == [1]
    assert 1.0000001 < shown <= 1.0000003


# --- sums and differences that could wrap around --------------------------------------------------------------------


def test_divergent_bar_nullable_columns_with_a_missing_category():
    d = pd.DataFrame({"cat": ["a", "b", "c"], "left": pd.array([3, None, 5], dtype="Int64"),
                      "right": pd.array([4.0, 2.0, 1.0], dtype="Float64")})
    table = vc.divergent_bar(d, "cat", "left", "right").table
    assert table["left"].dtype == float and np.isnan(table["left"][1]) and table["left"][[0, 2]].tolist() == [3, 5]
    assert table["right"].tolist() == [4.0, 2.0, 1.0]
    complete = vc.divergent_bar(d.fillna({"left": 0}), "cat", "left", "right").table
    assert complete["left"].dtype == np.int64


def test_divergent_bar_and_lollipop_int64_sums_do_not_wrap():
    v = np.array([2**62, 2**62, 5], dtype=np.int64)
    d = pd.DataFrame({"c": ["a", "a", "b"], "L": v, "R": [1, 1, 1]})
    assert vc.divergent_bar(d, "c", "L", "R").table["L"].tolist() == [2.0**63, 5.0]
    assert vc.lollipop(d, "c", "L", stat="sum").table["value"].tolist() == [2.0**63, 5.0]
    small = vc.lollipop(d.assign(L=[1, 2, 3]), "c", "L", stat="sum").table
    assert small["value"].dtype == np.int64 and small["value"].tolist() == [3, 3]


def test_safe_sum():
    assert safe_sum(pd.Series([1, 2], dtype="Int64")) == 3
    assert np.isnan(safe_sum(pd.Series([None], dtype="Int64"), min_count=1))
    assert safe_sum(pd.Series([2**62, 2**62])) == 2.0**63
    grouped = safe_sum(pd.Series([1, None, 2], dtype="Int64"), pd.Series(["x", "y", "x"]), min_count=1)
    assert grouped.dtype == float and grouped["x"] == 3 and np.isnan(grouped["y"])


def test_dumbbell_change_leaving_int64_is_a_float():
    d = pd.DataFrame({"l": ["a", "b"], "s": np.array([2**62, -2**62]), "e": np.array([2**62 + 1, 2**62])})
    table = vc.dumbbell(d, label="l", start="s", end="e").table
    assert table["l"].tolist() == ["a", "b"]  # b is the largest increase, so it is sorted last
    assert table["change"].tolist()[1] == 2.0**63 and table["pct_change"].tolist() == [0.0, 200.0]
    exact = vc.dumbbell(pd.DataFrame({"l": ["a"], "s": [2**62], "e": [2**62 + 1]}), "l", "s", "e").table
    assert exact["change"].dtype == np.int64 and exact["change"].tolist() == [1]


def test_dumbbell_percent_change_from_the_smallest_int64():
    d = pd.DataFrame({"l": ["a"], "s": np.array([-2**63]), "e": np.array([-2**63 + 2**61])})
    assert vc.dumbbell(d, "l", "s", "e").table["pct_change"].tolist() == [25.0]


# --- statistics at the edges ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 1e160, 1e-170])
def test_profile_boxes_eta_squared_does_not_depend_on_scale(scale):
    d = pd.DataFrame({"g": list("aabbcc") * 5, "y": np.arange(30.0) * scale})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eta = vc.profile_boxes(d, max_levels=3).table["eta_squared"].tolist()
    assert eta == pytest.approx([0.0355951])


@pytest.mark.parametrize("method", ["spearman", "pearson"])
@pytest.mark.parametrize("value", [np.inf, -np.inf, 1.0])
def test_an_infinite_constant_column_is_treated_as_constant(method, value):
    res = st.correlation_test([value] * 4, [1, 3, 2, 4], method)
    assert np.isnan(res["r"]) and np.isnan(res["p"]) and res["n"] == 4


def test_the_sturges_fallback_on_whole_numbers_is_documented_as_unrounded():
    x = np.array([0] * 80 + list(range(1, 21)))
    edges, rule = st._bin_edges(x, "fd")
    assert rule == "sturges (IQR is 0, so FD is undefined)"
    assert np.array_equal(edges, np.histogram_bin_edges(x, bins="sturges"))
    for doc in (st.histogram_bins.__doc__, vc.histogram.__doc__):
        assert "fractional width" in " ".join(doc.split())


def test_compare_correlations_without_any_defined_difference_does_not_warn():
    df = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 5, "x": [1] * 5 + [2] * 5, "y": np.arange(10.0)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = vc.compare_correlations(df, "g")
    assert r.table["difference"].isna().all()


def test_histogram_density_curve_matches_the_bars_when_edges_crop_the_data():
    x = pd.DataFrame({"x": np.linspace(0, 10, 2001)})
    for stat, height in (("density", 0.2), ("percent", 20.0), ("count", 200.0)):
        r = vc.histogram(x, "x", bins=[0, 1, 2, 3, 4, 5], stat=stat, density_curve=True)
        ax = r.axes[0, 0]
        assert r.info["outside_bins"] == 1000
        assert float(np.median(ax.lines[0].get_ydata())) == pytest.approx(height, rel=0.05)


def test_histogram_density_curve_is_skipped_when_every_value_is_outside():
    r = vc.histogram(pd.DataFrame({"x": np.arange(10.0)}), "x", bins=[20, 30], stat="density", density_curve=True)
    assert not r.axes[0, 0].lines


# --- columns labelled False / True ----------------------------------------------------------------------------------


def _boolean_labelled():
    trial = ds.trial()
    d = trial[["arm"]].copy()
    d[True] = trial["score"]
    d[False] = trial["baseline"]
    return d


def test_boolean_labelled_value_columns():
    d = _boolean_labelled()
    expected = d.groupby("arm")[True].mean()
    table = vc.estimation_plot(d, "arm", True, n_resamples=100).table
    assert table["mean"].tolist() == pytest.approx(expected.reindex(table["arm"]).tolist())
    assert vc.benchmark_bar(d, "arm", True).table["n"].sum() == d[True].notna().sum()
    assert vc.centered_bar(d, "arm", True).table["n"].sum() == d[True].notna().sum()
    assert vc.raincloud(d, "arm", True).table["n"].sum() == d[True].notna().sum()
    assert vc.ridgeplot(d, True, "arm").table["n"].sum() == d[True].notna().sum()
    assert vc.histogram(d, True, hue="arm").table["n"].sum() == d[True].notna().sum()
    table = vc.profile_boxes(d, categorical=["arm"], numeric=[True, False]).table
    assert sorted(table["numeric"].tolist()) == [False, True]


def test_boolean_labelled_group_column():
    m = ds.measurements()
    d = m[["length", "width", "depth", "mass"]].copy()
    d[True] = m["group"]
    r = vc.pca_plot(d, ["length", "width", "depth", "mass"], group=True)
    assert r.table[True].tolist() == m["group"].tolist()
