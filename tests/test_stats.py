"""Statistical helpers checked against SciPy and published worked examples."""

import inspect
import tracemalloc

import numpy as np
import pandas as pd
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


@pytest.mark.parametrize("method, n_resamples", [("BCa", 200), ("percentile", 5000)])
def test_bootstrap_memory_stays_bounded_for_large_samples(method, n_resamples):
    # Unbatched, SciPy held every resample at once (about 760 MiB traced here for the percentile interval at
    # 5,000 rows per sample) and BCa added several n x (n - 1) jackknife arrays (about 800 MiB with 200 resamples).
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 5000), rng.normal(0.1, 1, 5000)
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    before = tracemalloc.get_traced_memory()[0]
    try:
        est, lo, hi = st.bootstrap_ci(a, b, n_resamples=n_resamples, method=method)
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        if not was_tracing:
            tracemalloc.stop()
    assert lo < est < hi
    assert peak < 300 * 2**20


def test_bootstrap_up_to_2000_rows_matches_scipy_unbatched():
    # Batching changes how the seeded generator is consumed, so it starts only when some sample has more than
    # 2,000 rows: with every sample at or below that, intervals stay exactly what 1.1.0 reported. 3,000 resamples
    # is more than one batch at 2,000 rows (2**22 // 2000 = 2,097), so batching here would change the interval.
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 2000), rng.normal(0.1, 1, 1500)

    def diff(x, y):
        return float(np.mean(y) - np.mean(x))

    _, lo, hi = st.bootstrap_ci(a, b, statistic=diff, n_resamples=3000, seed=5)
    kwargs = dict(n_resamples=3000, confidence_level=0.95, method="BCa", vectorized=False)
    try:
        res = sps.bootstrap((a, b), diff, rng=np.random.default_rng(5), **kwargs)
    except TypeError:  # SciPy < 1.15 names the argument random_state
        res = sps.bootstrap((a, b), diff, random_state=np.random.default_rng(5), **kwargs)
    assert (lo, hi) == tuple(float(v) for v in res.confidence_interval)


def test_a_type_error_from_the_statistic_is_raised_once_without_rerunning_the_bootstrap():
    # The rng/random_state keyword used to be chosen by retrying on TypeError, which reran the whole bootstrap
    # when the user's statistic raised one and chained the same error to itself.
    a, b = [1.0, 2.0, 4.0, 7.0], [3.0, 5.0, 6.0, 9.0, 10.0]

    def failing(limit):
        calls = []

        def statistic(x, y):
            calls.append(1)
            if len(calls) > limit:
                raise TypeError("statistic failed on one resample")
            return float(np.mean(y) - np.mean(x))
        return statistic, calls

    keyword = "rng" if "rng" in inspect.signature(sps.bootstrap).parameters else "random_state"  # SciPy < 1.15
    statistic, direct = failing(5)
    with pytest.raises(TypeError, match="failed on one resample"):
        sps.bootstrap((a, b), statistic, n_resamples=50, vectorized=False, **{keyword: np.random.default_rng(0)})
    statistic, calls = failing(5)
    with pytest.raises(TypeError, match="failed on one resample") as info:
        st.bootstrap_ci(a, b, statistic=statistic, n_resamples=50)
    assert len(calls) == len(direct) and info.value.__context__ is None


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


def _largest_remainder_with_fractions(values, total):
    """Hamilton's method on Fractions, the reference for the integer arithmetic of largest_remainder."""
    from fractions import Fraction

    parts = [Fraction(v) for v in values]
    whole = sum(parts)
    if whole == 0:
        return [0] * len(parts)
    quotas = [p * total / whole for p in parts]
    base = [q.numerator // q.denominator for q in quotas]
    for i in sorted(range(len(parts)), key=lambda i: base[i] - quotas[i])[:total - sum(base)]:
        base[i] += 1
    return base


@pytest.mark.parametrize("seed", range(6))
def test_largest_remainder_matches_exact_fractions(seed):
    rng = np.random.default_rng(seed)
    for _ in range(150):
        n = int(rng.integers(1, 25))
        values = [rng.random(n), rng.integers(0, 6, n) / 4, rng.gamma(2, 10, n) * 10.0 ** rng.integers(-300, 300),
                  np.where(rng.random(n) < 0.3, 0.0, rng.random(n) * 10.0 ** rng.integers(-320, 300, n))][seed % 4]
        total = int(rng.choice([0, 1, 7, 100, 10**18, 2**70]))
        assert st.largest_remainder(values, total).tolist() == _largest_remainder_with_fractions(values.tolist(), total)


@pytest.mark.parametrize("values", [np.array([10**16 + 1, 10**16 + 3]), [10**16 + 1, 10**16 + 3],
                                    pd.Series([10**16 + 1, 10**16 + 3], dtype="Int64"),
                                    np.array([10**16 + 1, 10**16 + 3], dtype=np.uint64)],
                         ids=["int64", "list", "Int64", "uint64"])
def test_largest_remainder_keeps_integers_above_2_53_exact(values):
    # As floats both counts round to 1e16 and the larger one took a unit from the smaller.
    assert st.largest_remainder(values, 2 * 10**16 + 4).tolist() == [10**16 + 1, 10**16 + 3]


def test_largest_remainder_takes_python_ints_beyond_int64():
    assert st.largest_remainder([10**20 + 1, 10**20 + 3], 2 * 10**20 + 4).tolist() == [10**20 + 1, 10**20 + 3]
    with pytest.raises(ValueError, match="non-negative"):
        st.largest_remainder([10**20, -1], 10)


def test_largest_remainder_serves_equal_remainders_in_input_order():
    assert st.largest_remainder([1, 1, 1], 2).tolist() == [1, 1, 0]
    assert st.largest_remainder([1, 1, 1, 1], 6).tolist() == [2, 2, 1, 1]  # the sorted branch for many leftovers
    assert st.largest_remainder(np.ones(40), 1).tolist() == [1] + [0] * 39  # and the heap branch for few


def test_largest_remainder_is_fast_for_a_million_values():
    import time

    values = np.random.default_rng(0).gamma(2, 10, 10**6)
    start = time.perf_counter()
    out = st.largest_remainder(values, 100)
    assert out.sum() == 100
    assert time.perf_counter() - start < 8  # 15 s or more with a Fraction per value


@pytest.mark.parametrize("r", [1.5, -2.0, np.inf, -np.inf])
def test_compare_correlations_test_rejects_values_that_are_not_correlations(r):
    # 1.5 was clipped to 0.999999 and gave z = 48 and p = 0
    with pytest.raises(ValueError, match=r"correlations must be between -1 and 1, got \["):
        st.compare_correlations_test(r, 100, 0.3, 100)
    with pytest.raises(ValueError, match="between -1 and 1"):
        st.compare_correlations_test(np.array([0.2, 0.3]), 100, np.array([0.1, r]), 100)


def test_compare_correlations_test_accepts_the_whole_range_and_missing_values():
    z, p = st.compare_correlations_test(np.array([1.0, -1.0, 1 + 2e-16, np.nan]), 50, 0.5, 50)
    assert np.isfinite(z[:3]).all() and np.isnan(z[3]) and np.isnan(p[3])
    assert z[0] == z[2]  # a perfect correlation that rounding left a hair above 1 is still clipped
