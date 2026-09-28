"""Statistical helpers checked against SciPy and published worked examples."""

import numpy as np
import pytest
from scipy import stats as sps

from viz_calc import stats as st


def test_wilson_matches_reference_values():
    # Brown, Cai & DasGupta (2001) style checks; values agree with statsmodels' proportion_confint(method="wilson").
    lo, hi = st.wilson_ci(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(0.27753, abs=1e-4)
    lo, hi = st.wilson_ci(5, 10)
    assert (lo, hi) == pytest.approx((0.23659, 0.76341), abs=1e-4)


def test_wilson_handles_zero_n_and_arrays():
    lo, hi = st.wilson_ci(np.array([0, 3]), np.array([0, 4]))
    assert np.isnan(lo[0]) and np.isnan(hi[0])
    assert 0 < lo[1] < 0.75 < hi[1] <= 1


def test_mean_ci():
    x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    m, lo, hi = st.mean_ci(x)
    se = np.std(x, ddof=1) / np.sqrt(len(x))
    t = sps.t.ppf(0.975, len(x) - 1)
    assert m == pytest.approx(5.0)
    assert (lo, hi) == pytest.approx((5 - t * se, 5 + t * se))
    assert np.isnan(st.mean_ci([1.0])[1])


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 20), rng.normal(0.8, 3, 35)
    res = st.welch_test(a, b)
    ref = sps.ttest_ind(b, a, equal_var=False)
    assert res["t"] == pytest.approx(ref.statistic)
    assert res["p"] == pytest.approx(ref.pvalue)
    assert res["difference"] == pytest.approx(b.mean() - a.mean())
    ci = ref.confidence_interval(0.95)
    assert (res["ci_low"], res["ci_high"]) == pytest.approx((ci.low, ci.high))


def test_hedges_g():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a + 2
    res = st.hedges_g(a, b)
    d = 2 / np.std(a, ddof=1)
    assert res["g"] == pytest.approx(d * (1 - 3 / (4 * 8 - 1)))
    assert res["ci_low"] < res["g"] < res["ci_high"]


def test_bootstrap_is_reproducible_and_sensible():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 60), rng.normal(1, 1, 60)
    r1 = st.bootstrap_ci(a, b, n_resamples=2000, seed=3)
    r2 = st.bootstrap_ci(a, b, n_resamples=2000, seed=3)
    assert r1 == r2
    est, lo, hi = r1
    assert lo < est < hi
    assert est == pytest.approx(b.mean() - a.mean())


def test_adjust_pvalues_worked_example():
    p = [0.01, 0.04, 0.03, 0.005]
    assert st.adjust_pvalues(p, "holm") == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert st.adjust_pvalues(p, "fdr_bh") == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert st.adjust_pvalues(p, "bonferroni") == pytest.approx([0.04, 0.16, 0.12, 0.02])
    assert st.adjust_pvalues(p, "none") == pytest.approx(p)


def test_adjust_pvalues_matches_scipy_bh_and_ignores_nan():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 0.2, 30)
    assert st.adjust_pvalues(p, "fdr_bh") == pytest.approx(sps.false_discovery_control(p, method="bh"))
    out = st.adjust_pvalues([0.01, np.nan, 0.02], "holm")
    assert np.isnan(out[1])
    assert out[[0, 2]] == pytest.approx([0.02, 0.02])


def test_correlation_test_ci():
    rng = np.random.default_rng(4)
    x = rng.normal(size=50)
    y = 0.5 * x + rng.normal(size=50)
    res = st.correlation_test(x, y)
    r = sps.pearsonr(x, y)[0]
    half = 1.959964 / np.sqrt(47)
    assert res["r"] == pytest.approx(r)
    assert (res["ci_low"], res["ci_high"]) == pytest.approx((np.tanh(np.arctanh(r) - half), np.tanh(np.arctanh(r) + half)), abs=1e-6)
    assert res["n"] == 50


def test_correlation_drops_incomplete_pairs():
    res = st.correlation_test([1, 2, 3, np.nan, 5, 6], [2, 4, 5, 9, np.nan, 12])
    assert res["n"] == 4


def test_compare_correlations_textbook_example():
    # r1=.50 vs r2=.30 with n=100 each: z ≈ 1.67, p ≈ .095
    z, p = st.compare_correlations_test(0.5, 100, 0.3, 100)
    assert float(z) == pytest.approx(1.670, abs=1e-3)
    assert float(p) == pytest.approx(0.0949, abs=1e-3)
    z_s, _ = st.compare_correlations_test(0.5, 100, 0.3, 100, method="spearman")
    assert float(z_s) == pytest.approx(1.670 / np.sqrt(1.06), abs=1e-3)


def test_histogram_bins_fd_width():
    x = np.arange(1000, dtype=float)
    edges = st.histogram_bins(x, "fd")
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    assert np.diff(edges)[0] == pytest.approx(2 * iqr / 1000 ** (1 / 3), rel=0.05)


@pytest.mark.parametrize("values,total", [([1, 1, 1], 100), ([5, 3, 2, 1], 7), ([0.4, 0.4, 0.2], 10), ([9, 1], 1)])
def test_largest_remainder_always_sums_to_total(values, total):
    out = st.largest_remainder(values, total)
    assert out.sum() == total
    assert (out >= 0).all()


def test_largest_remainder_example():
    assert list(st.largest_remainder([1, 1, 1], 100)) == [34, 33, 33]
    assert list(st.largest_remainder([0, 0], 10)) == [0, 0]
    with pytest.raises(ValueError):
        st.largest_remainder([-1, 2], 10)
