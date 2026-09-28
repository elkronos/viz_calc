"""Distributions: histograms, ridgelines and raincloud plots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

from . import stats as st
from ._core import VizResult, category_order, check_choice, check_dataframe, check_numeric, get_ax, palette

__all__ = ["histogram", "ridgeplot", "raincloud"]


def _describe(s: pd.Series) -> dict[str, float]:
    s = s.dropna()
    return {"n": int(s.size), "mean": s.mean(), "median": s.median(), "sd": s.std(ddof=1),
            "q1": s.quantile(0.25), "q3": s.quantile(0.75)}


def _kde(values: np.ndarray, grid: np.ndarray, bw_method: Any = None) -> np.ndarray | None:
    values = values[np.isfinite(values)]
    if values.size < 2 or np.ptp(values) == 0:
        return None
    return gaussian_kde(values, bw_method=bw_method)(grid)


def histogram(
    data: pd.DataFrame,
    x: str,
    hue: str | None = None,
    facet: str | None = None,
    bins: int | Sequence[float] | Literal["fd", "scott", "sturges", "auto"] = "fd",
    stat: Literal["count", "density", "percent"] = "count",
    density_curve: bool = False,
    ref_line: Literal["mean", "median"] | float | None = None,
    hue_order: Sequence[Any] | None = None,
    facet_order: Sequence[Any] | None = None,
    col_wrap: int = 3,
    colors: Sequence[str] | str | None = None,
    alpha: float = 0.55,
) -> VizResult:
    """Histogram with principled, shared bins, optionally by group and facet.

    Bin edges come from a published rule — Freedman–Diaconis by default,
    which is robust to outliers — and are computed once on all of *x*, so
    every group and facet uses the same bins and can be compared directly.
    Use ``stat="density"`` or ``"percent"`` when groups differ in size.

    ``ref_line`` draws the mean or median **of each facet** (and of each hue
    group within it), not a single global value.

    If the Freedman–Diaconis width is unusable (the IQR is zero, e.g. heavily
    tied or zero-inflated data, or FD would need more than 100,000 bins)
    Sturges' rule is used instead; ``info["bin_rule"]`` says which rule was
    applied and why.

    References
    ----------
    Freedman, D., & Diaconis, P. (1981). *Z. Wahrscheinlichkeitstheorie verw.
    Gebiete*, 57, 453–476.
    """
    check_dataframe(data, [x, hue, facet])
    check_numeric(data, x)
    check_choice("stat", stat, ["count", "density", "percent"])
    values = data[x].dropna().to_numpy(float)
    if isinstance(bins, str):
        edges, rule = st._bin_edges(values, bins)
    else:
        edges, rule = np.histogram_bin_edges(values, bins=bins), "user"

    facets = category_order(data[facet], facet_order) if facet is not None else [None]
    hues = category_order(data[hue], hue_order) if hue is not None else [None]
    cols = palette(len(hues), colors)
    ncol = min(col_wrap, len(facets))
    nrow = int(np.ceil(len(facets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False, sharex=True, sharey=True)
    grid = np.linspace(edges[0], edges[-1], 256)
    width = np.diff(edges)
    rows = []
    for ax, f in zip(axes.flat, facets):
        fdata = data if f is None else data[data[facet] == f]
        for h, col in zip(hues, cols):
            s = fdata[x] if h is None else fdata.loc[fdata[hue] == h, x]
            s = s.dropna().to_numpy(float)
            counts, _ = np.histogram(s, bins=edges)
            if stat == "density":
                heights = counts / (counts.sum() * width) if counts.sum() else counts.astype(float)
            elif stat == "percent":
                heights = 100 * counts / counts.sum() if counts.sum() else counts.astype(float)
            else:
                heights = counts
            if h is None:
                ax.bar(edges[:-1], heights, width=width, align="edge", color=col, alpha=alpha,
                       edgecolor="white", lw=0.5)
            else:  # outlined steps keep overlapping groups readable
                ax.stairs(heights, edges, fill=True, color=col, alpha=alpha * 0.4)
                ax.stairs(heights, edges, color=col, lw=1.8, label=str(h))
            if density_curve:
                dens = _kde(s, grid)
                if dens is not None:
                    scale = {"density": 1.0, "percent": 100 * np.mean(width), "count": s.size * np.mean(width)}[stat]
                    ax.plot(grid, dens * scale, color=col, lw=1.5)
            if ref_line is not None and s.size:
                v = {"mean": np.mean, "median": np.median}[ref_line](s) if isinstance(ref_line, str) else float(ref_line)
                ax.axvline(v, color=col if h is not None else "black", ls="--", lw=1.2)
            rows.append({"facet": f, "hue": h, **_describe(pd.Series(s))})
        if f is not None:
            ax.set_title(f"{facet} = {f}", fontsize="medium", loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(facets):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(x)
    for ax in axes[:, 0]:
        ax.set_ylabel({"count": "Count", "density": "Density", "percent": "Percent"}[stat])
    if hue is not None:
        axes.flat[0].legend(title=str(hue), frameon=False)
    fig.suptitle(f"Distribution of {x}  (bins: {rule}, width ≈ {np.mean(width):.3g})", x=0.01, ha="left")
    fig.tight_layout()
    table = pd.DataFrame(rows).drop(columns=[c for c, used in (("facet", facet), ("hue", hue)) if used is None])
    return VizResult(fig, axes, table, {"bin_edges": edges, "bin_rule": rule, "stat": stat})


def ridgeplot(
    data: pd.DataFrame,
    x: str,
    group: str,
    order: Sequence[Any] | None = None,
    overlap: float = 0.6,
    bw_method: Any = None,
    show_median: bool = True,
    colors: Sequence[str] | str | None = "viridis",
    ax: Axes | None = None,
) -> VizResult:
    """Stacked, overlapping density curves ("joyplot") for many groups.

    Useful when there are too many groups for overlaid histograms. All
    curves share one x-axis and a common height scale, so peak heights are
    comparable. The median of each group is marked.
    """
    check_dataframe(data, [x, group])
    check_numeric(data, x)
    groups = category_order(data[group], order)
    values = data[x].dropna().to_numpy(float)
    pad = 0.05 * np.ptp(values) if np.ptp(values) else 1.0
    grid = np.linspace(values.min() - pad, values.max() + pad, 400)
    dens = {g: _kde(data.loc[data[group] == g, x].dropna().to_numpy(float), grid, bw_method) for g in groups}
    peak = max((d.max() for d in dens.values() if d is not None), default=1.0)
    cols = palette(len(groups), colors)
    fig, ax = get_ax(ax, figsize=(8, 0.7 * len(groups) + 2))
    step = 1 - overlap
    rows = []
    for i, g in enumerate(groups):
        base = (len(groups) - 1 - i) * step
        d = dens[g]
        s = data.loc[data[group] == g, x]
        if d is not None:
            y = base + d / peak
            ax.fill_between(grid, base, y, color=cols[i], alpha=0.8, zorder=2 * i + 1, lw=0)
            ax.plot(grid, y, color="white", lw=1.2, zorder=2 * i + 2)
            if show_median:
                m = s.median()
                ax.plot([m, m], [base, base + np.interp(m, grid, d) / peak], color="black", lw=1, zorder=2 * i + 2)
        rows.append({group: g, **_describe(s)})
    ax.set_yticks([(len(groups) - 1 - i) * step for i in range(len(groups))], [str(g) for g in groups])
    ax.set_xlabel(x)
    ax.set_ylabel(group)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return VizResult(fig, ax, pd.DataFrame(rows))


def raincloud(
    data: pd.DataFrame,
    x: str,
    y: str,
    order: Sequence[Any] | None = None,
    colors: Sequence[str] | str | None = None,
    bw_method: Any = None,
    seed: int = 0,
    ax: Axes | None = None,
) -> VizResult:
    """Raincloud plot: half-violin density, box plot and raw data together.

    A box plot alone hides sample size and multimodality; a violin alone
    hides the data. Raincloud plots show all three with little clutter
    (Allen et al., 2019). Groups run along the vertical axis.

    References
    ----------
    Allen, M., Poggiali, D., Whitaker, K., Marshall, T. R., & Kievit, R. A.
    (2019). Raincloud plots: a multi-platform tool for robust data
    visualization. *Wellcome Open Research*, 4, 63.
    """
    check_dataframe(data, [x, y])
    check_numeric(data, y)
    groups = category_order(data[x], order)
    cols = palette(len(groups), colors)
    rng = np.random.default_rng(seed)
    fig, ax = get_ax(ax, figsize=(8, 1.4 * len(groups) + 1.5))
    rows = []
    for i, g in enumerate(groups):
        s = data.loc[data[x] == g, y].dropna().to_numpy(float)
        pos = len(groups) - 1 - i
        if s.size:
            # Each group gets its own grid over its own range, so the density never suggests unseen values
            # and narrow groups are still drawn in full detail.
            g_grid = np.linspace(s.min(), s.max(), 300)
            d = _kde(s, g_grid, bw_method)
            if d is not None:
                ax.fill_between(g_grid, pos + 0.1, pos + 0.1 + 0.45 * d / d.max(), color=cols[i], alpha=0.6, lw=0)
        if s.size:
            box = dict(positions=[pos], widths=0.12, showfliers=False, patch_artist=True, manage_ticks=False,
                       boxprops={"facecolor": "white", "edgecolor": "black"}, medianprops={"color": "black"})
            try:
                ax.boxplot(s, orientation="horizontal", **box)
            except TypeError:  # Matplotlib < 3.10
                ax.boxplot(s, vert=False, **box)
            ax.scatter(s, pos - 0.22 + rng.uniform(-0.08, 0.08, s.size), s=10, color=cols[i], alpha=0.6, lw=0)
        rows.append({x: g, **_describe(pd.Series(s))})
    ax.set_yticks([len(groups) - 1 - i for i in range(len(groups))], [f"{g}\n(n={r['n']})" for g, r in zip(groups, rows)])
    ax.set_xlabel(y)
    ax.set_ylabel(x)
    ax.spines[["top", "right"]].set_visible(False)
    return VizResult(fig, ax, pd.DataFrame(rows))
