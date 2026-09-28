"""Multivariate views: principal components and radar profiles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from ._core import NEUTRAL, VizResult, category_order, check_choice, check_dataframe, check_numeric, get_ax, palette

__all__ = ["pca", "pca_plot", "radar"]


def pca(data: pd.DataFrame, features: Sequence[str], scale: bool = True) -> dict[str, Any]:
    """Principal component analysis by singular value decomposition.

    Rows with a missing feature are dropped. With ``scale=True`` features are
    standardized first (a correlation-matrix PCA), which is the right default
    when features are on different units; otherwise large-variance features
    dominate (Jolliffe & Cadima, 2016). Component signs are fixed so that the
    largest-magnitude loading of each component is positive, making results
    reproducible.

    Returns a dict with ``scores`` (DataFrame, index aligned to *data*),
    ``loadings`` (features × components), ``explained_variance_ratio`` and
    ``eigenvalues``.

    References
    ----------
    Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a
    review and recent developments. *Phil. Trans. R. Soc. A*, 374, 20150202.
    """
    check_dataframe(data, features)
    check_numeric(data, *features)
    X = data[list(features)].dropna()
    if len(X) < 3:
        raise ValueError("need at least three complete rows")
    Z = X - X.mean()
    if scale:
        sd = X.std(ddof=1).replace(0, np.nan)
        if sd.isna().any():
            raise ValueError(f"constant feature(s) cannot be scaled: {list(sd[sd.isna()].index)}")
        Z = Z / sd
    U, S, Vt = np.linalg.svd(Z.to_numpy(float), full_matrices=False)
    signs = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])
    Vt *= signs[:, None]
    U *= signs[None, :]
    eig = S**2 / (len(X) - 1)
    names = [f"PC{i + 1}" for i in range(len(S))]
    return {
        "scores": pd.DataFrame(U * S, index=X.index, columns=names),
        "loadings": pd.DataFrame(Vt.T, index=list(features), columns=names),
        "explained_variance_ratio": pd.Series(eig / eig.sum(), index=names),
        "eigenvalues": pd.Series(eig, index=names),
    }


def _ellipse(ax: Axes, pts: np.ndarray, level: float, kind: str, color: str) -> dict[str, float] | None:
    if len(pts) < 3:
        return None
    cov = np.cov(pts, rowvar=False)
    if kind == "confidence":
        cov = cov / len(pts)  # covariance of the mean
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    radius = np.sqrt(chi2.ppf(level, df=2))
    width, height = 2 * radius * np.sqrt(np.clip(vals, 0, None))
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    centre = pts.mean(axis=0)
    ax.add_patch(Ellipse(centre, width, height, angle=angle, facecolor=color, alpha=0.12, edgecolor=color, lw=1.5))
    return {"cx": centre[0], "cy": centre[1], "width": width, "height": height, "angle": angle}


def pca_plot(
    data: pd.DataFrame,
    features: Sequence[str],
    group: str | None = None,
    components: tuple[int, int] = (1, 2),
    scale: bool = True,
    ellipse: Literal["data", "confidence"] | None = "data",
    level: float = 0.95,
    loadings: int | bool = 0,
    order: Sequence[Any] | None = None,
    colors: Sequence[str] | str | None = None,
    ax: Axes | None = None,
) -> VizResult:
    """Scores on two principal components, with group ellipses and optional biplot arrows.

    Axis labels give the share of variance each component explains.

    Parameters
    ----------
    components
        1-based component numbers to plot, e.g. ``(1, 2)`` or ``(2, 3)``.
    ellipse
        ``"data"``: a normal-theory region expected to contain *level* of each
        group's points (radius ``sqrt(chi2_2(level))`` in the eigenbasis of the
        group covariance). ``"confidence"``: a *level* confidence region for the
        group **mean**. ``None``: no ellipse.
    loadings
        Draw arrows for the *n* features with the largest loadings on the
        plotted components (``True`` = all features).
    """
    check_dataframe(data, [*features, group])
    check_choice("ellipse", ellipse, ["data", "confidence", None])
    res = pca(data, features, scale)
    ci, cj = components
    k = len(res["explained_variance_ratio"])
    if not (1 <= ci <= k and 1 <= cj <= k) or ci == cj:
        raise ValueError(f"components must be two different numbers between 1 and {k}")
    pc_x, pc_y = f"PC{ci}", f"PC{cj}"
    scores = res["scores"][[pc_x, pc_y]].copy()
    groups = [None]
    if group:
        complete = data[list(features)].notna().all(axis=1)  # the rows pca() kept, matched by position
        scores[group] = data.loc[complete, group].array  # positional, and keeps a Categorical's order
        groups = category_order(scores[group], order)
    cols = palette(len(groups), colors)

    fig, ax = get_ax(ax, figsize=(8, 7))
    ellipses = {}
    for g, c in zip(groups, cols):
        sub = scores if g is None else scores[scores[group] == g]
        ax.scatter(sub[pc_x], sub[pc_y], s=22, color=c, alpha=0.75, lw=0, label=None if g is None else str(g))
        if ellipse:
            ellipses[g] = _ellipse(ax, sub[[pc_x, pc_y]].to_numpy(float), level, ellipse, c)
    if loadings:
        L = res["loadings"][[pc_x, pc_y]]
        n = len(L) if loadings is True else int(loadings)
        top = L.pow(2).sum(axis=1).sort_values(ascending=False).index[:n]
        span = np.abs(scores[[pc_x, pc_y]].to_numpy()).max()
        mult = 0.8 * span / np.abs(L.loc[top].to_numpy()).max()
        for f in top:
            lx, ly = L.loc[f] * mult
            ax.annotate("", (lx, ly), (0, 0), arrowprops={"arrowstyle": "->", "color": NEUTRAL, "lw": 1.2})
            ax.text(lx * 1.08, ly * 1.08, f, color="black", fontsize=8, ha="center", va="center")
    evr = res["explained_variance_ratio"]
    ax.set_xlabel(f"{pc_x} ({evr[pc_x]:.1%} of variance)")
    ax.set_ylabel(f"{pc_y} ({evr[pc_y]:.1%} of variance)")
    ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
    ax.axvline(0, color="#dddddd", lw=0.8, zorder=0)
    ax.set_aspect("equal", adjustable="datalim")
    if group:
        ax.legend(title=group, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    title = "standardized features" if scale else "unscaled features"
    if ellipse:
        title += f"; ellipses: {level:.0%} {'of data' if ellipse == 'data' else 'CI of mean'}"
    ax.set_title(f"PCA ({title})", loc="left", fontsize="medium")
    ax.spines[["top", "right"]].set_visible(False)
    return VizResult(fig, ax, scores.reset_index(names="row"),
                     {**res, "ellipses": ellipses, "components": (pc_x, pc_y)})


def radar(
    data: pd.DataFrame,
    metrics: Sequence[str],
    group: str,
    stat: Literal["mean", "median"] = "mean",
    normalize: Literal["data", "groups", "none"] = "data",
    order: Sequence[Any] | None = None,
    colors: Sequence[str] | str | None = None,
    ax: Axes | None = None,
) -> VizResult:
    """Radar (spider) chart comparing group profiles across several metrics.

    Rows are first aggregated to one value per group and metric (*stat*).
    Because metrics usually have different units, each axis is rescaled:

    * ``"data"`` (default): 0 = smallest and 1 = largest **observed value**
      of the metric, so group summaries sit inside the range of the data.
    * ``"groups"``: 0 = lowest group, 1 = highest group. Exaggerates small
      differences and always puts one group at the centre.
    * ``"none"``: raw values (only sensible when metrics share a unit).

    A group with no data on a metric is left as a gap on that axis. Raw and
    scaled values are in the returned table. Radar charts are best
    for spotting profile *shapes*; the enclosed area depends on the order of
    the metrics and should not be compared.
    """
    check_dataframe(data, [*metrics, group])
    check_numeric(data, *metrics)
    check_choice("normalize", normalize, ["data", "groups", "none"])
    check_choice("stat", stat, ["mean", "median"])
    if len(metrics) < 3:
        raise ValueError("radar needs at least three metrics")
    groups = category_order(data[group], order)
    raw = data.groupby(group, observed=True)[list(metrics)].agg(stat).reindex(groups)
    raw = pd.DataFrame(raw.to_numpy(dtype=float, na_value=np.nan), index=raw.index, columns=raw.columns)
    scaled = raw.copy()
    if normalize != "none":
        ref = data[list(metrics)].astype(float) if normalize == "data" else raw
        lo, span = ref.min(), (ref.max() - ref.min()).replace(0, np.nan)
        scaled = (raw - lo) / span
        flat = [m for m in metrics if pd.isna(span[m])]  # metric has one value only: put groups mid-scale
        for m in flat:
            scaled[m] = raw[m].where(raw[m].isna(), 0.5)  # a group with no data stays missing (a gap)
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    closed = np.r_[angles, angles[0]]
    cols = palette(len(groups), colors)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    else:
        fig = ax.figure
    for g, c in zip(groups, cols):
        vals = scaled.loc[g].to_numpy(float)
        # Markers keep a group visible even when missing metrics break its outline into isolated points.
        ax.plot(closed, np.r_[vals, vals[0]], color=c, lw=2, marker="o", ms=4, label=str(g))
        if not np.isnan(vals).any():  # a polygon with a missing vertex would be misleading
            ax.fill(closed, np.r_[vals, vals[0]], color=c, alpha=0.12)
    ax.set_xticks(angles, list(metrics))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    if normalize != "none":
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1], ["min", "", "max"], color=NEUTRAL, fontsize=8)
        ax.set_title(f"{stat} per group; axes scaled to the {'observed' if normalize == 'data' else 'group'} range",
                     fontsize="medium", color=NEUTRAL, pad=20)
    ax.legend(title=group, frameon=False, loc="upper left", bbox_to_anchor=(1.05, 1.05))
    table = raw.add_suffix(f"_{stat}").join(scaled.add_suffix("_scaled")).rename_axis(group).reset_index()
    return VizResult(fig, ax, table, {"normalize": normalize})
