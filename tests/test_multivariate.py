"""PCA scaling and the group ellipses of pca_plot."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

import viz_calc as vc
from viz_calc import multivariate

# --- the confidence ellipse is Hotelling's T² region for the group mean ------------------------------------------


def _inside(e, point):
    ang = np.deg2rad(e["angle"])
    rot = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
    u = rot @ (np.asarray(point) - [e["cx"], e["cy"]])
    return (u[0] / (e["width"] / 2)) ** 2 + (u[1] / (e["height"] / 2)) ** 2 <= 1


def _small_groups():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(size=(15, 3)), columns=["a", "b", "c"])
    df["g"] = ["p"] * 5 + ["q"] * 10
    return df


def test_confidence_ellipse_of_a_small_group_has_the_hotelling_size():
    df = _small_groups()
    res = vc.pca_plot(df, ["a", "b", "c"], group="g", ellipse="confidence")
    for g, n in (("p", 5), ("q", 10)):
        pts = res.table.loc[res.table["g"] == g, ["PC1", "PC2"]].to_numpy()
        axes = np.sqrt(np.sort(np.linalg.eigvalsh(np.cov(pts, rowvar=False) / n))[::-1])
        radius = np.sqrt(2 * (n - 1) / (n - 2) * sps.f.ppf(0.95, 2, n - 2))
        e = res.info["ellipses"][g]
        assert (e["width"], e["height"]) == pytest.approx(tuple(2 * radius * axes), rel=1e-12)


def test_data_ellipse_keeps_the_chi_square_radius():
    df = _small_groups()
    res = vc.pca_plot(df, ["a", "b", "c"], group="g", ellipse="data")
    pts = res.table.loc[res.table["g"] == "p", ["PC1", "PC2"]].to_numpy()
    axes = np.sqrt(np.sort(np.linalg.eigvalsh(np.cov(pts, rowvar=False)))[::-1])
    e = res.info["ellipses"]["p"]
    assert (e["width"], e["height"]) == pytest.approx(tuple(2 * np.sqrt(sps.chi2.ppf(0.95, 2)) * axes), rel=1e-12)


@pytest.mark.parametrize("n", [3, 5, 10])
def test_confidence_ellipse_covers_the_true_mean_at_its_level(n):
    # The chi-square radius covered the mean of a correlated normal 51% (n = 3), 75% (n = 5) and 87% (n = 10)
    # of the time for a nominal 95%.
    rng = np.random.default_rng(n)
    ax = SimpleNamespace(add_patch=lambda patch: None)  # only the geometry is needed; drawing 2000 patches is slow
    reps = 2000
    hits = sum(
        _inside(multivariate._ellipse(ax, rng.multivariate_normal([0, 0], [[4, 1.5], [1.5, 1]], n), 0.95,
                                      "confidence", "k"), [0, 0])
        for _ in range(reps)
    )
    assert hits / reps == pytest.approx(0.95, abs=0.02)
