"""Constant groups, p-value ranges, pd.NA in plain lists and FD bin limits in viz_calc.stats."""

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
