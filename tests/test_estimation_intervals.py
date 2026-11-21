"""estimation_plot's interval for the mean difference: the Welch interval beside the bootstrap one, ci=, small groups."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets
from viz_calc import stats as st

# Three per group: the BCa interval excludes zero while the Welch p is 0.12.
A, B = [-0.455, -0.992, 0.06], [2.34, 0.508, 0.38]
SMALL = pd.DataFrame({"g": ["A"] * 3 + ["B"] * 3, "y": A + B})


def _drawn_intervals(ax):
    """``(low, high)`` of each error bar in the lower panel, left to right."""
    out = []
    for container in ax.containers:
        (x0, low), (_, high) = container.lines[2][0].get_segments()[0]
        out.append((x0, min(low, high), max(low, high)))
    return [(low, high) for _, low, high in sorted(out)]


def test_welch_interval_is_reported_beside_the_unchanged_bca_interval():
    res = vc.estimation_plot(SMALL, "g", "y")
    row = res.info["comparisons"].iloc[0]
    w = st.welch_test(A, B)
    assert row[["welch_ci_low", "welch_ci_high"]].tolist() == pytest.approx([w["ci_low"], w["ci_high"]], rel=1e-12)
    # The BCa interval is the default and keeps its value: it excludes zero, the Welch interval does not.
    assert row[["ci_low", "ci_high"]].tolist() == pytest.approx([0.663333, 2.974], abs=1e-6)
    assert row["welch_ci_low"] < 0 < row["welch_ci_high"] and row["p"] == pytest.approx(0.120073, abs=1e-6)
    assert res.info["ci_method"] == "bca"
    assert _drawn_intervals(res.axes[1]) == [pytest.approx((row["ci_low"], row["ci_high"]), rel=1e-12)]
    columns = list(res.info["comparisons"].columns)
    assert columns[columns.index("ci_high") + 1: columns.index("ci_high") + 3] == ["welch_ci_low", "welch_ci_high"]


@pytest.mark.parametrize("seed", range(12))
def test_welch_interval_agrees_with_the_unadjusted_p(seed):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"g": ["a"] * 6 + ["b"] * 6 + ["c"] * 9, "y": np.r_[rng.normal(0, 1, 6), rng.normal(1.2, 1, 6),
                                                                        rng.lognormal(0, 1, 9)]})
    comps = vc.estimation_plot(d, "g", "y", n_resamples=200, p_adjust="none").info["comparisons"]
    excludes = (comps["welch_ci_low"] > 0) | (comps["welch_ci_high"] < 0)
    assert (excludes == (comps["p"] < 0.05)).all()


@pytest.mark.parametrize("ci,title", [("welch", "Mean difference with Welch t 95% CI"),
                                      ("percentile", "Mean difference with bootstrap 95% CI (percentile)")])
def test_ci_sets_the_reported_and_drawn_interval_and_the_title(ci, title):
    t = datasets.trial()
    res = vc.estimation_plot(t, "arm", "score", n_resamples=500, ci=ci)
    comps = res.info["comparisons"].set_index("group")
    control = t.loc[t["arm"] == "control", "score"]
    for g in ["low dose", "high dose"]:
        other = t.loc[t["arm"] == g, "score"]
        if ci == "welch":
            w = st.welch_test(control, other)
            want = (w["ci_low"], w["ci_high"])
        else:
            want = st.bootstrap_ci(control, other, n_resamples=500, method="percentile", seed=0)[1:]
        assert comps.loc[g, ["ci_low", "ci_high"]].tolist() == pytest.approx(want, rel=1e-12)
        w = st.welch_test(control, other)
        assert comps.loc[g, ["welch_ci_low", "welch_ci_high"]].tolist() == pytest.approx([w["ci_low"], w["ci_high"]],
                                                                                        rel=1e-12)
    assert _drawn_intervals(res.axes[1]) == [pytest.approx(tuple(comps.loc[g, ["ci_low", "ci_high"]]), rel=1e-12)
                                             for g in ["low dose", "high dose"]]
    assert res.axes[1].get_title(loc="left") == title
    assert res.info["ci_method"] == ci
    bca = vc.estimation_plot(t, "arm", "score", n_resamples=500).info["comparisons"]
    keep = ["difference", "hedges_g", "g_ci_low", "g_ci_high", "welch_t", "df", "p", "p_adjusted"]
    pd.testing.assert_frame_equal(res.info["comparisons"][keep], bca[keep])


def test_ci_welch_keeps_the_bootstrap_distribution():
    shaded = [vc.estimation_plot(SMALL, "g", "y", n_resamples=500, ci=ci).axes[1].collections[0].get_paths()[0].vertices
              for ci in ("bca", "welch")]
    np.testing.assert_array_equal(*shaded)


def test_small_groups_are_listed_and_named_without_a_warning():
    rng = np.random.default_rng(3)
    d = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 30 + ["c"] * 14 + ["d"] * 15, "y": rng.normal(size=63)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = vc.estimation_plot(d, "g", "y", reference="b", n_resamples=200)
    assert res.info["small_groups"] == {"a": 4, "c": 14}
    assert res.axes[1].get_title(loc="left").split("\n") == [
        "Mean difference with bootstrap 95% CI (BCa)", "Bootstrap CIs tend to be too narrow for n < 15 (a, c)"]
    welch = vc.estimation_plot(d, "g", "y", reference="b", n_resamples=200, ci="welch")
    assert welch.info["small_groups"] == {"a": 4, "c": 14}
    assert "\n" not in welch.axes[1].get_title(loc="left")
    many = pd.DataFrame({"g": np.repeat(list("pqrs"), 3), "y": rng.normal(size=12)})
    title = vc.estimation_plot(many, "g", "y", n_resamples=200, ci="percentile").axes[1].get_title(loc="left")
    assert title.endswith("\nBootstrap CIs tend to be too narrow for n < 15 (4 groups)")
    big = vc.estimation_plot(datasets.trial(), "arm", "score", n_resamples=200)
    assert big.info["small_groups"] == {}
    assert big.axes[1].get_title(loc="left") == "Mean difference with bootstrap 95% CI (BCa)"


def test_unknown_ci_is_rejected_before_drawing():
    before = plt.get_fignums()
    with pytest.raises(ValueError, match=r"ci must be one of \['bca', 'welch', 'percentile'\], got 'BCa'"):
        vc.estimation_plot(SMALL, "g", "y", ci="BCa")
    assert plt.get_fignums() == before


def test_docs_give_the_small_sample_caveat_and_the_welch_interval():
    root = Path(__file__).resolve().parents[1]
    methodology = " ".join((root / "docs" / "methodology.md").read_text("utf-8").split())
    walkthrough = " ".join((root / "docs" / "walkthroughs" / "comparing-groups.md").read_text("utf-8").split())
    assert "fewer than 15 values are listed" in methodology and "`welch_ci_low`" in methodology
    assert 'info["small_groups"]' in walkthrough and 'ci="welch"' in walkthrough
    comps = vc.estimation_plot(datasets.trial(), "arm", "score", n_resamples=200).info["comparisons"]
    low, high = (comps.set_index("group").loc[g, ["welch_ci_low", "welch_ci_high"]] for g in ["low dose", "high dose"])
    shown = (f"{low.iloc[0]:.1f} to {low.iloc[1]:.1f} for low dose and {high.iloc[0]:.1f} to {high.iloc[1]:.1f} "
             "for high dose").replace("-", "−")
    assert shown in walkthrough
