"""PCA scaling and the group ellipses of pca_plot."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

import viz_calc as vc
from viz_calc import datasets, multivariate

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


# --- PCA input: constant features and Arrow-backed columns --------------------------------------------------------

FEATURES = ["length", "width", "depth", "mass"]


@pytest.mark.parametrize("c", [0.1, 1 / 3, 0.5, 7.0])
def test_pca_rejects_a_constant_feature_whatever_its_binary_value(c):
    # The float mean of 0.1 or 1/3 is off by an ulp, so the SD came out near 3e-17 instead of 0 and the constant
    # column was standardized into a component holding 20% of the variance.
    m = datasets.measurements().assign(c=c)
    with pytest.raises(ValueError, match=r"constant feature\(s\) cannot be scaled: \['c'\]"):
        vc.pca(m, [*FEATURES, "c"])
    with pytest.raises(ValueError, match=r"constant feature\(s\) cannot be scaled: \['c'\]"):
        vc.pca_plot(m, [*FEATURES, "c"], group="group")


@pytest.mark.parametrize("c", [0.1, 1 / 3])
def test_unscaled_pca_gives_a_constant_feature_no_variance(c):
    m = datasets.measurements()
    got = vc.pca(m.assign(c=c), [*FEATURES, "c"], scale=False)
    dyadic = vc.pca(m.assign(c=0.5), [*FEATURES, "c"], scale=False)
    assert got["eigenvalues"].iloc[-1] == 0.0
    pd.testing.assert_series_equal(got["explained_variance_ratio"], dyadic["explained_variance_ratio"])


@pytest.mark.parametrize("offset", [1e12, 2.0**52, -(2.0**52)])
def test_pca_does_not_change_with_an_exact_offset(offset):
    # Centring on a mean rounded at 2**52 moved the explained variance ratios by 1.7e-4.
    m = datasets.measurements()
    m["length"] = np.round(m["length"] * 10) * 2  # even whole numbers, exact at these offsets
    want, got = vc.pca(m, FEATURES), vc.pca(m.assign(length=m["length"] + offset), FEATURES)
    for key in ("explained_variance_ratio", "eigenvalues"):
        assert got[key].to_numpy() == pytest.approx(want[key].to_numpy(), rel=1e-12)
    assert got["scores"].to_numpy() == pytest.approx(want["scores"].to_numpy(), rel=1e-12, abs=1e-12)


def test_pca_of_ordinary_features_is_unchanged():
    m = datasets.measurements()
    m.loc[3, "width"] = np.nan
    X = m[FEATURES].dropna()
    Z = ((X - X.mean()) / X.std(ddof=1)).to_numpy(float)
    S = np.linalg.svd(Z, full_matrices=False)[1]
    assert vc.pca(m, FEATURES)["eigenvalues"].to_numpy().tolist() == (S**2 / (len(X) - 1)).tolist()


@pytest.mark.parametrize("backend", ["pyarrow", "numpy_nullable"])
def test_pca_accepts_arrow_backed_and_nullable_features(backend):
    # pandas 2.0 raised NotImplementedError ('ArrowExtensionArray does not support reshape') when centring.
    if backend == "pyarrow":
        pytest.importorskip("pyarrow")
    m = datasets.measurements()
    converted = m.assign(**m[FEATURES].convert_dtypes(dtype_backend=backend))
    assert all(isinstance(t, pd.api.extensions.ExtensionDtype) for t in converted[FEATURES].dtypes)
    want, got = vc.pca(m, FEATURES), vc.pca(converted, FEATURES)
    for key in ("scores", "loadings"):
        pd.testing.assert_frame_equal(got[key], want[key])
    pd.testing.assert_series_equal(got["eigenvalues"], want["eigenvalues"])
    pd.testing.assert_frame_equal(vc.pca_plot(converted, FEATURES, group="group").table,
                                  vc.pca_plot(m, FEATURES, group="group").table)
