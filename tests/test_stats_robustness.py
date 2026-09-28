"""Statistical helpers on small, nullable, constant, extreme and integer-valued input."""

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

import viz_calc as vc
from viz_calc import datasets
from viz_calc import stats as st

# --- nullable dtypes with pd.NA (pandas 2.0 cannot convert them with np.asarray(dtype=float)) -------------------


def _nullable_pair():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.integers(0, 50, 30).astype(float), "b": rng.normal(size=30), "c": rng.normal(size=30)})
    df.loc[3, "a"] = np.nan
    df.loc[7, "b"] = np.nan
    nullable = df.astype({"a": "Int64", "b": "Float64"})
    assert nullable["a"].isna().sum() == 1 and nullable["a"].dtype == "Int64"
    return df, nullable


@pytest.mark.parametrize("func", [
    lambda d: st.mean_ci(d["a"]),
    lambda d: st.welch_test(d["a"], d["b"]),
    lambda d: st.hedges_g(d["a"], d["b"]),
    lambda d: st.bootstrap_ci(d["a"], d["b"], n_resamples=300),
    lambda d: st.correlation_test(d["a"], d["b"]),
    lambda d: st.correlation_test(d["a"], d["b"], "spearman"),
    lambda d: tuple(st.histogram_bins(d["b"])),
    lambda d: tuple(st.adjust_pvalues(d["b"].abs() / 10)),
    lambda d: st.wilson_ci(d["a"].iloc[:3], d["a"].iloc[3:6] + 60),
    lambda d: st.compare_correlations_test(d["b"].iloc[5:9], d["a"].iloc[:4] + 10, 0.1, 40),
])
def test_stats_treat_pd_na_like_nan(func):
    df, nullable = _nullable_pair()
    np.testing.assert_equal(func(nullable), func(df))


def test_largest_remainder_reports_pd_na_as_missing():
    with pytest.raises(ValueError, match="not missing"):
        st.largest_remainder(pd.array([1, None, 2], dtype="Int64"), 10)
    assert list(st.largest_remainder(pd.array([1, 1, 2], dtype="Int64"), 4)) == [1, 1, 2]


def test_correlation_charts_accept_nullable_columns_with_na():
    meas = datasets.measurements()
    meas.loc[[3, 50, 120], "width"] = np.nan
    nullable = meas.convert_dtypes()  # Float64 columns; the NaNs become pd.NA
    assert nullable["width"].isna().sum() == 3
    for call in (lambda d: vc.correlogram(d, method="spearman").table,
                 lambda d: vc.compare_correlations(d, group="group").table,
                 lambda d: vc.profile_scatters(d, fit=False).table,
                 lambda d: pd.DataFrame([vc.quadrant_plot(d, "length", "width").info["correlation"]])):
        pd.testing.assert_frame_equal(call(nullable), call(meas), check_dtype=False)
    assert vc.correlogram(nullable).table.loc[lambda t: t.var2 == "width", "n"].eq(147).all()


# --- exact Spearman p-values for small n ---------------------------------------------------------------------------


def _brute_force_p(x, y):
    """Share of all orderings of y whose |Spearman rho| with x reaches the observed one (float arithmetic)."""
    rx, ry = sps.rankdata(x), sps.rankdata(y)
    shuffled = ry[np.array(list(itertools.permutations(range(len(y)))))]
    rxc = rx - rx.mean()
    null = (shuffled - shuffled.mean(axis=1, keepdims=True)) @ rxc / (np.linalg.norm(rxc) * np.linalg.norm(ry - ry.mean()))
    return float(np.mean(np.abs(null) >= abs(sps.spearmanr(x, y)[0]) - 1e-12))


@pytest.mark.parametrize("n", range(3, 10))
def test_perfect_spearman_has_exact_p_of_two_over_n_factorial(n):
    x = np.arange(1.0, n + 1)
    res = st.correlation_test(x, x**2, "spearman")
    assert res["p"] == pytest.approx(2 / math.factorial(n), rel=1e-12)
    assert np.isnan(res["ci_low"]) and np.isnan(res["ci_high"])  # |rho| = 1, even when rounded to 0.9999999999999999


def test_small_n_spearman_p_matches_full_enumeration_with_ties():
    rng = np.random.default_rng(3)
    checked = 0
    for n in range(3, 8):
        for k in range(12):
            x = rng.integers(0, 3 if k % 2 else 50, n).astype(float)
            y = rng.integers(0, 4 if k % 3 == 0 else 50, n).astype(float)
            if np.ptp(x) == 0 or np.ptp(y) == 0:
                continue
            assert st.correlation_test(x, y, "spearman")["p"] == pytest.approx(_brute_force_p(x, y), abs=1e-12)
            checked += 1
    assert checked > 40


def test_spearman_from_ten_pairs_and_pearson_keep_the_t_test():
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=10), rng.normal(size=10)
    assert st.correlation_test(x, y, "spearman")["p"] == pytest.approx(sps.spearmanr(x, y)[1])
    assert st.correlation_test(x[:5], y[:5])["p"] == pytest.approx(sps.pearsonr(x[:5], y[:5])[1])


def test_tiny_spearman_pairs_are_not_starred():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 45], "c": [3, 1, 4, 2]})
    table = vc.correlogram(df, method="spearman").table.set_index(["var1", "var2"])
    assert table.loc[("a", "b"), "p"] == pytest.approx(1 / 12)
    assert not table["significant"].any()
    ps = vc.profile_scatters(df, method="spearman", min_levels=2, fit=False).table
    assert ps.loc[(ps.x == "a") & (ps.y == "b"), "p"].item() == pytest.approx(1 / 12)


# --- level validation -------------------------------------------------------------------------------------------------

BAD_LEVELS = [95, 1, 0, -0.5, 1.5, np.nan, "0.95", None]


@pytest.mark.parametrize("level", BAD_LEVELS)
@pytest.mark.parametrize("func", [
    lambda lv: st.mean_ci([1.0, 2.0, 3.0], lv),
    lambda lv: st.mean_ci([], lv),
    lambda lv: st.wilson_ci(3, 10, lv),
    lambda lv: st.welch_test([1.0, 2, 3], [2.0, 3, 5], lv),
    lambda lv: st.hedges_g([1.0, 2, 3], [2.0, 3, 5], lv),
    lambda lv: st.bootstrap_ci([1.0, 2, 3], [2.0, 3, 5], level=lv, n_resamples=100),
    lambda lv: st.correlation_test([1.0, 2, 3, 4, 5], [2.0, 1, 4, 3, 5], level=lv),
])
def test_stats_reject_levels_outside_zero_one(func, level):
    with pytest.raises(ValueError, match="level must be strictly between 0 and 1"):
        func(level)


@pytest.mark.parametrize("func", [
    lambda d, lv: vc.benchmark_bar(d, x="arm", y="score", threshold=50, level=lv),
    lambda d, lv: vc.estimation_plot(d, x="arm", y="score", level=lv, n_resamples=200),
    lambda d, lv: vc.centered_bar(d, x="arm", y="score", level=lv),
    lambda d, lv: vc.percent_grid(d, column="improved", level=lv),
])
def test_charts_reject_a_percentage_level(func):
    trial = datasets.trial()
    with pytest.raises(ValueError, match="got 95"):
        func(trial, 95)
    func(trial, np.float32(0.9))  # any real number in (0, 1) is fine


# --- zero-variance and extreme-magnitude groups ------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # the BCa bootstrap of constant groups is degenerate
def test_welch_t_has_the_sign_of_the_difference_for_constant_groups():
    assert st.welch_test([2, 2, 2], [1, 1, 1])["t"] == -np.inf
    assert st.welch_test([1, 1, 1], [2, 2, 2])["t"] == np.inf
    df = pd.DataFrame({"g": ["ref"] * 3 + ["trt"] * 3, "y": [2.0] * 3 + [1.0] * 3})
    comps = vc.estimation_plot(df, "g", "y", n_resamples=200).info["comparisons"]
    assert comps["welch_t"].item() == -np.inf and comps["difference"].item() == -1.0


@pytest.mark.parametrize("a,b", [(0.1, 0.2), (0.3, 0.1), (1 / 3, 2 / 3), (1.0, 2.0)])
def test_constant_groups_are_constant_whatever_their_binary_representation(a, b):
    w = st.welch_test([a] * 3, [b] * 3)
    assert w["t"] == np.copysign(np.inf, b - a) and w["df"] == 4.0 and w["p"] == 0.0
    assert w["ci_low"] == w["ci_high"] == w["difference"]
    assert np.isnan(st.hedges_g([a] * 3, [b] * 3)["g"])
    m, lo, hi = st.mean_ci([a] * 3)
    assert lo == hi == m


@pytest.mark.parametrize("scale", [1e160, 1e-160, 1e300])
def test_effect_sizes_and_tests_do_not_depend_on_scale(scale):
    a, b = np.array([1.0, 2, 3, 4]), np.array([2.0, 3, 5, 7])
    g, w, m = st.hedges_g(a, b), st.welch_test(a, b), st.mean_ci(a)
    gs, ws, ms = st.hedges_g(a * scale, b * scale), st.welch_test(a * scale, b * scale), st.mean_ci(a * scale)
    assert gs["g"] == pytest.approx(g["g"]) and gs["ci_low"] == pytest.approx(g["ci_low"])
    assert (ws["t"], ws["df"], ws["p"]) == pytest.approx((w["t"], w["df"], w["p"]))
    assert (ws["ci_low"], ws["ci_high"]) == pytest.approx((w["ci_low"] * scale, w["ci_high"] * scale))
    assert ms == pytest.approx(tuple(v * scale for v in m))


def test_power_of_two_rescaling_leaves_ordinary_results_bit_identical():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 20), rng.normal(0.8, 3, 35)
    w = st.welch_test(a, b)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    assert w["t"] == (b.mean() - a.mean()) / np.sqrt(va + vb)
    assert st.mean_ci(a)[0] == a.mean()
    assert st.welch_test(a * 2.0**600, b * 2.0**600)["t"] == w["t"]


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")  # waffle's own percentages
def test_largest_remainder_fills_the_grid_when_the_sum_overflows():
    assert list(st.largest_remainder([1e308, 1e308], 100)) == [50, 50]
    assert list(st.largest_remainder([1.5e308, 7.5e307, 7.5e307], 8)) == [4, 2, 2]
    tiles = vc.waffle(pd.DataFrame({"c": ["a", "b"], "v": [1e308, 1e308]}), "c", "v").table["tiles"]
    assert tiles.tolist() == [50, 50]


# --- Freedman-Diaconis bins on whole numbers ---------------------------------------------------------------------


def test_fd_bins_on_counts_have_whole_widths_and_no_empty_comb():
    rng = np.random.default_rng(0)
    counts = rng.poisson(3, 1000)
    edges, rule = st._bin_edges(counts, "fd")
    assert rule == "fd (whole-number widths for integer data)"
    assert np.all(np.diff(edges) == 1.0) and np.all(edges % 1 == 0.5)
    hist, _ = np.histogram(counts, edges)
    assert hist.tolist() == np.bincount(counts).tolist()
    res = vc.histogram(pd.DataFrame({"visits": pd.array(counts, dtype="Int64")}), x="visits")
    assert np.array_equal(res.info["bin_edges"], edges) and res.info["bin_rule"] == rule


def test_fd_width_is_rounded_to_a_whole_number_on_wide_integer_data():
    rng = np.random.default_rng(2)
    x = rng.integers(-500, 500, 300)
    edges = st.histogram_bins(x)
    fd = 2 * np.subtract(*np.percentile(x, [75, 25])) / 300 ** (1 / 3)
    width = np.diff(edges)
    assert np.all(width == round(fd)) and edges[0] == x.min() - 0.5 and edges[-1] >= x.max() + 0.5
    assert np.all(np.histogram(x, edges)[0] > 0)


def test_fd_bins_unchanged_for_non_integer_or_huge_values():
    rng = np.random.default_rng(4)
    x = rng.normal(size=500)
    assert np.array_equal(st.histogram_bins(x), np.histogram_bin_edges(x, bins="fd"))
    big = 2.0**60 + rng.integers(0, 1000, 200) * 2048.0  # whole numbers too large for half-integer edges
    edges, rule = st._bin_edges(big, "fd")
    assert rule == "fd" and np.array_equal(edges, np.histogram_bin_edges(big, bins="fd"))
    assert np.array_equal(st.histogram_bins(np.arange(20), "auto"), np.histogram_bin_edges(np.arange(20), "auto"))


# --- option validation ---------------------------------------------------------------------------------------------


def test_adjust_pvalues_rejects_unknown_method_even_without_p_values():
    for p in ([0.01, 0.2], [np.nan], []):
        with pytest.raises(ValueError, match=r"method must be one of \['holm', 'fdr_bh', 'bonferroni', 'none'\]"):
            st.adjust_pvalues(p, "bh")


@pytest.mark.parametrize("func", [
    lambda d: vc.correlogram(d, p_adjust="bh"),
    lambda d: vc.compare_correlations(d, group="group", p_adjust="bh"),
])
def test_correlation_charts_name_the_bad_p_adjust(func):
    with pytest.raises(ValueError, match=r"p_adjust must be one of \['holm', 'fdr_bh', 'bonferroni', 'none'\]"):
        func(datasets.measurements())


def test_correlation_method_error_lists_valid_values():
    with pytest.raises(ValueError, match=r"method must be one of \['pearson', 'spearman'\], got 'kendall'"):
        st.correlation_test([1, 2, 3], [3, 1, 2], "kendall")
