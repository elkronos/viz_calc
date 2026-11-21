"""Constant groups, p-value ranges, pd.NA in plain lists and FD bin limits in viz_calc.stats."""

import re

import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import stats as st


@pytest.mark.parametrize("v,n1,n2", [(0.1, 3, 10), (0.3, 2, 10), (0.7, 2, 3), (1.1, 2, 6), (1.0, 3, 10)])
def test_identical_constant_groups_do_not_differ(v, n1, n2):
    w = st.welch_test([v] * n1, [v] * n2)
    assert w["difference"] == w["ci_low"] == w["ci_high"] == 0.0
    assert np.isnan(w["t"]) and w["p"] == 1.0
    assert np.isnan(st.hedges_g([v] * n1, [v] * n2)["g"])


def test_different_constant_groups_keep_an_infinite_t():
    w = st.welch_test([0.1] * 3, [0.2] * 10)
    assert w["t"] == np.inf and w["p"] == 0.0 and w["difference"] == 0.2 - 0.1


@pytest.mark.parametrize("v", [0.1, 0.3, 1 / 3, 1e300])
def test_mean_ci_of_a_constant_sample_is_its_value(v):
    assert st.mean_ci([v] * 3) == (v, v, v)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # the BCa bootstrap of constant groups is degenerate
def test_estimation_plot_does_not_call_identical_constant_groups_different():
    df = pd.DataFrame({"g": ["a"] * 2 + ["b"] * 3 + ["c"] * 4, "y": [0.1] * 5 + [0.5, 0.5, 0.6, 0.6]})
    comps = vc.estimation_plot(df, "g", "y", n_resamples=200).info["comparisons"].set_index("group")
    assert np.isnan(comps.loc["b", "welch_t"]) and comps.loc["b", "p"] == comps.loc["b", "p_adjusted"] == 1.0


@pytest.mark.parametrize("method", ["holm", "fdr_bh", "bonferroni", "none"])
@pytest.mark.parametrize("p", [[-0.5, 0.2], [0.01, 3.0], [0.2, np.inf]])
def test_adjust_pvalues_rejects_values_outside_the_unit_interval(p, method):
    with pytest.raises(ValueError, match="between 0 and 1"):
        st.adjust_pvalues(p, method)


def test_adjust_pvalues_accepts_the_bounds_and_missing_values():
    out = st.adjust_pvalues([0.0, 1.0, np.nan], "holm")
    assert out[0] == 0.0 and out[1] == 1.0 and np.isnan(out[2])


@pytest.mark.parametrize("wrap", [list, lambda v: np.array(v, dtype=object)])
def test_pd_na_in_plain_sequences_counts_as_missing(wrap):
    assert st.mean_ci(wrap([1.0, pd.NA, 3.0])) == st.mean_ci([1.0, 3.0])
    assert st.adjust_pvalues(wrap([0.01, pd.NA, 0.04]), "none")[[0, 2]].tolist() == [0.01, 0.04]
    r = st.correlation_test(wrap([1.0, 2, pd.NA, 4, 5]), [2.0, 1, 3, 5, 4])
    assert r["n"] == 4
    lo, hi = st.wilson_ci(wrap([1, pd.NA]), [2, 4])
    assert np.isnan(lo[1]) and np.isnan(hi[1]) and 0 < lo[0] < hi[0] < 1


def test_non_numeric_objects_still_raise():
    with pytest.raises(TypeError):
        st.mean_ci([1.0, object(), 3.0])


def test_whole_number_rounding_respects_the_bin_limit():
    rng = np.random.default_rng(0)
    x = np.r_[rng.integers(0, 14, 999), 130_000].astype(float)
    edges, rule = st._bin_edges(x, "fd")
    assert len(edges) - 1 <= 100_000 and rule == "sturges (FD would need 130,001 bins)"
    assert len(st.histogram_bins(x)) - 1 <= 100_000


def test_whole_number_rounding_keeps_fd_below_the_limit():
    x = np.r_[np.tile(np.arange(13.0), 80)[:999], 60_000.0]
    edges, rule = st._bin_edges(x, "fd")
    assert rule == "fd (whole-number widths for integer data)" and len(edges) - 1 <= 100_000


def test_fd_keeps_a_real_iqr_at_large_magnitudes():
    x = 10.0**15 + np.array([0, 1, 3, 7, 2, 20])
    assert vc.histogram(pd.DataFrame({"x": x}), "x").info["bin_rule"] == "fd (whole-number widths for integer data)"
    edges, rule = st._bin_edges(x + 0.5, "auto")  # same data off the integer grid
    assert not rule.startswith("sturges")


# --- wilson_ci on impossible counts -------------------------------------------------------------------------------


@pytest.mark.parametrize("k,n", [(5, 3), (-1, 10), (10.5, 10), (-0.5, 10), (3, -5), (5, np.inf), (np.inf, np.inf),
                                 (np.nan, np.inf), (1, 0)])
def test_wilson_ci_rejects_impossible_counts(k, n):
    with pytest.raises(ValueError, match=r"successes <= n, got successes="):
        st.wilson_ci(k, n)


def test_wilson_ci_names_the_impossible_entries_of_an_array():
    with pytest.raises(ValueError, match=r"got successes=4, n=3$"):
        st.wilson_ci(np.array([1, 4, np.nan]), np.array([3, 3, 3]))


@pytest.mark.parametrize("k,n,shown", [(1_000_001, 1_000_000, "successes=1000001, n=1000000"),
                                       (0.1 + 0.2, 0.3, "successes=0.30000000000000004, n=0.3"),
                                       (5, np.inf, "successes=5, n=inf"), (-2.5, 1e300, "successes=-2.5, n=1e+300")])
def test_wilson_ci_error_tells_the_counts_apart(k, n, shown):
    # Six significant digits printed 1000001 of 1000000 as "successes=1e+06, n=1e+06".
    with pytest.raises(ValueError, match=f"got {re.escape(shown)}$"):
        st.wilson_ci(k, n)


def test_wilson_ci_keeps_nan_for_missing_or_zero_counts_and_accepts_weighted_counts():
    assert all(np.isnan(v) for v in st.wilson_ci(np.nan, 10) + st.wilson_ci(0, 0) + st.wilson_ci(3, np.nan))
    lo, hi = st.wilson_ci(2.5, 10)
    assert 0 < lo < 0.25 < hi < 1


def test_wilson_ci_of_huge_counts_does_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert st.wilson_ci(1e200, 1e200) == (1.0, 1.0)
        lo, hi = st.wilson_ci(0, 1e200)
        assert lo == 0.0 and 0 < hi < 1e-199


# --- bootstrap_ci: default statistic for one sample, checked options ----------------------------------------------


def test_bootstrap_ci_of_one_sample_bootstraps_its_mean():
    x = [1.0, 2.0, 4.0, 7.0]
    assert st.bootstrap_ci(x, n_resamples=500) == st.bootstrap_ci(x, statistic=np.mean, n_resamples=500)
    assert st.bootstrap_ci(x, n_resamples=500)[0] == 3.5


@pytest.mark.parametrize("count", [0, 3])
def test_bootstrap_ci_default_statistic_names_the_sample_count(count):
    with pytest.raises(ValueError, match=f"one sample .* or two .*got {count}; pass statistic="):
        st.bootstrap_ci(*[[1.0, 2.0, 4.0]] * count)


@pytest.mark.parametrize("data", [([1.0, 2.0, 4.0, 7.0], [3.0, 5.0, 6.0, 9.0, 10.0]), ([1.0, 1.0], [1.0, 1.0])],
                         ids=["varied", "constant"])
@pytest.mark.parametrize("kwargs, message", [
    (dict(method="nope"), r"method must be one of \['BCa', 'percentile', 'basic'\], got 'nope'"),
    (dict(method=None), r"method must be one of .*got None"),
    (dict(n_resamples=0), "n_resamples must be a whole number of at least 1, got 0"),
    (dict(n_resamples=True), "n_resamples must be a whole number of at least 1, got True"),
    (dict(seed=-1), "seed must be a whole number of at least 0, got -1"),
    (dict(seed=True), "seed must be a whole number of at least 0, got True"),
    (dict(seed="x"), "seed must be a whole number of at least 0, got 'x'"),
], ids=["method", "method-none", "n_resamples-0", "n_resamples-bool", "seed-negative", "seed-bool", "seed-text"])
def test_bootstrap_ci_checks_its_options_before_resampling(data, kwargs, message):
    # The constant pair skips SciPy entirely, so it used to accept any option.
    with pytest.raises(ValueError, match=message):
        st.bootstrap_ci(*data, **kwargs)


def test_bootstrap_ci_method_keeps_any_letter_case():
    a, b = [1.0, 2.0, 4.0, 7.0], [3.0, 5.0, 6.0, 9.0, 10.0]
    assert st.bootstrap_ci(a, b, method="bca", n_resamples=300) == st.bootstrap_ci(a, b, n_resamples=300)
    assert st.bootstrap_ci(a, b, method="Percentile", n_resamples=300) == st.bootstrap_ci(
        a, b, method="percentile", n_resamples=300)
