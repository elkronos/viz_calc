"""Distributions: histograms, ridgelines and raincloud plots."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

from . import stats as st
from ._core import (
    VizResult,
    category_order,
    check_choice,
    check_count,
    check_dataframe,
    check_has_values,
    check_not_reserved,
    check_numeric,
    check_range,
    cleanup_on_error,
    get_ax,
    level_label,
    palette,
)

__all__ = ["histogram", "ridgeplot", "raincloud"]


_SUMMARY = ["n", "mean", "median", "sd", "q1", "q3"]  # the columns _describe adds to a table


def _describe(s: pd.Series) -> dict[str, float]:
    s = s.dropna()
    return {"n": int(s.size), "mean": s.mean(), "median": s.median(), "sd": st._sd(s.to_numpy(float)),
            "q1": s.quantile(0.25), "q3": s.quantile(0.75)}


def _kde(values: np.ndarray, grid: np.ndarray, bw_method: Any = None) -> np.ndarray | None:
    values = values[np.isfinite(values)]
    if values.size < 2 or np.ptp(values) == 0:
        return None
    s = st._unit_scale(values)  # an exact power of two, so the KDE's covariance neither overflows nor underflows
    return gaussian_kde(values / s, bw_method=bw_method)(grid / s) / s


@cleanup_on_error
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
    group within it), not a single global value; a number draws a fixed line
    at that value.

    *bins* is one of the rules ``"fd"``, ``"scott"``, ``"sturges"`` and
    ``"auto"``, a number of equal-width bins (an ``int``), or a sequence of
    increasing bin edges. Values outside given edges are not drawn:
    ``stat="percent"`` and ``"density"`` are computed over the values inside
    the edges, ``info["outside_bins"]`` counts the values left out, and the
    title says how many there are. The table's summaries use every value.
    The *density_curve* is fitted to every value and scaled to the bars, so
    for ``"density"`` and ``"percent"`` it too is relative to the values
    inside the edges.

    If the Freedman–Diaconis width is unusable (the IQR is zero, e.g. heavily
    tied or zero-inflated data, or FD would need more than 100,000 bins)
    Sturges' rule is used instead; ``info["bin_rule"]`` says which rule was
    applied and why.

    On whole-number data (counts, scores, integer dtypes) ``bins="fd"``
    rounds the Freedman–Diaconis width to a whole number (at least 1) and
    puts the edges on half-integers, starting at ``min(x) - 0.5``. Every
    value then sits inside a bin, away from its edges, and every bin spans
    the same number of possible values, so there is no comb of empty bars;
    ``info["bin_rule"]`` reads ``"fd (whole-number widths for integer
    data)"``. The Sturges bins that replace unusable FD bins are not
    rounded, so on zero-inflated counts they can have a fractional width.
    The other rules are used as published (see
    :func:`viz_calc.stats.histogram_bins`).

    References
    ----------
    Freedman, D., & Diaconis, P. (1981). *Z. Wahrscheinlichkeitstheorie verw.
    Gebiete*, 57, 453–476.
    """
    check_dataframe(data, [x, hue, facet])
    check_numeric(data, x)
    check_has_values(data, x)
    check_choice("stat", stat, ["count", "density", "percent"])
    rules = ["fd", "scott", "sturges", "auto"]
    if isinstance(bins, str):
        check_choice("bins", bins, rules)
    elif np.ndim(bins) == 0:
        if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError(f"bins must be one of {rules}, a positive int or a sequence of increasing edges, "
                             f"got {bins!r}")
    else:
        try:
            given = st._as_float(bins)
        except (TypeError, ValueError):
            given = np.array([np.nan])
        if given.ndim != 1 or given.size < 2 or not np.isfinite(given).all() or (np.diff(given) <= 0).any():
            raise ValueError(f"bins must be one of {rules}, a positive int or a sequence of at least two finite, "
                             f"increasing edges, got {bins!r}")
    if isinstance(ref_line, str):
        check_choice("ref_line", ref_line, ["mean", "median"])
    elif ref_line is not None and (isinstance(ref_line, bool) or not isinstance(ref_line, Real)):
        raise ValueError(f"ref_line must be 'mean', 'median', a number or None, got {ref_line!r}")
    check_count("col_wrap", col_wrap)
    check_range("alpha", alpha, 0, 1)
    values = data[x].dropna().to_numpy(float)
    if isinstance(bins, str):
        edges, rule = st._bin_edges(values, bins)
    else:
        edges, rule = np.histogram_bin_edges(values, bins=bins), "user"
    outside = int(((values < edges[0]) | (values > edges[-1])).sum())  # only given edges can leave values out

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
            s = fdata[x] if h is None else fdata[x][fdata[hue] == h]
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
                ax.stairs(heights, edges, color=col, lw=1.8, label=level_label(h))
            if density_curve and counts.sum():
                dens = _kde(s, grid)
                if dens is not None:
                    # The KDE spreads over every value; density and percent bars count only those inside the edges.
                    ratio = s.size / counts.sum()
                    scale = {"density": ratio, "percent": 100 * np.mean(width) * ratio,
                             "count": s.size * np.mean(width)}[stat]
                    ax.plot(grid, dens * scale, color=col, lw=1.5)
            if ref_line is not None and s.size:
                v = {"mean": np.mean, "median": np.median}[ref_line](s) if isinstance(ref_line, str) else float(ref_line)
                ax.axvline(v, color=col if h is not None else "black", ls="--", lw=1.2)
            rows.append({"facet": f, "hue": h, **_describe(pd.Series(s))})
        if f is not None:
            ax.set_title(f"{facet} = {level_label(f)}", fontsize="medium", loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(facets):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(x)
    for ax in axes[:, 0]:
        ax.set_ylabel({"count": "Count", "density": "Density", "percent": "Percent"}[stat])
    if hue is not None:
        axes.flat[0].legend(title=str(hue), frameon=False)
    note = f"; {outside} value{'s' if outside > 1 else ''} outside the bins not shown" if outside else ""
    fig.suptitle(f"Distribution of {x}  (bins: {rule}, width ≈ {np.mean(width):.3g}{note})", x=0.01, ha="left")
    fig.tight_layout()
    table = pd.DataFrame(rows).drop(columns=[c for c, used in (("facet", facet), ("hue", hue)) if used is None])
    return VizResult(fig, axes, table, {"bin_edges": edges, "bin_rule": rule, "stat": stat, "outside_bins": outside})


@cleanup_on_error
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

    Parameters
    ----------
    overlap
        How far each curve reaches into the row above, as a share of the
        tallest peak: from 0 (no overlap) up to, but not including, 1.
    """
    check_dataframe(data, [x, group])
    check_not_reserved(_SUMMARY, group=group)
    check_numeric(data, x)
    check_range("overlap", overlap, 0, 1, include_high=False)
    check_has_values(data, x)
    groups = category_order(data[group], order)
    values = data[x].dropna().to_numpy(float)
    pad = 0.05 * np.ptp(values) if np.ptp(values) else 1.0
    grid = np.linspace(values.min() - pad, values.max() + pad, 400)
    dens = {g: _kde(data[x][data[group] == g].dropna().to_numpy(float), grid, bw_method) for g in groups}
    peak = max((d.max() for d in dens.values() if d is not None), default=1.0)
    cols = palette(len(groups), colors)
    fig, ax = get_ax(ax, figsize=(8, 0.7 * len(groups) + 2))
    step = 1 - overlap
    rows = []
    for i, g in enumerate(groups):
        base = (len(groups) - 1 - i) * step
        d = dens[g]
        s = data[x][data[group] == g]
        if d is not None:
            y = base + d / peak
            ax.fill_between(grid, base, y, color=cols[i], alpha=0.8, zorder=2 * i + 1, lw=0)
            ax.plot(grid, y, color="white", lw=1.2, zorder=2 * i + 2)
            if show_median:
                m = s.median()
                ax.plot([m, m], [base, base + np.interp(m, grid, d) / peak], color="black", lw=1, zorder=2 * i + 2)
        rows.append({group: g, **_describe(s)})
    ax.set_yticks([(len(groups) - 1 - i) * step for i in range(len(groups))], [level_label(g) for g in groups])
    ax.set_xlabel(x)
    ax.set_ylabel(group)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return VizResult(fig, ax, pd.DataFrame(rows))


@cleanup_on_error
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
    check_not_reserved(_SUMMARY, x=x)
    check_numeric(data, y)
    check_has_values(data, y)
    groups = category_order(data[x], order)
    cols = palette(len(groups), colors)
    rng = np.random.default_rng(seed)
    fig, ax = get_ax(ax, figsize=(8, 1.4 * len(groups) + 1.5))
    rows = []
    for i, g in enumerate(groups):
        s = data[y][data[x] == g].dropna().to_numpy(float)
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
    ax.set_yticks([len(groups) - 1 - i for i in range(len(groups))], [f"{level_label(g)}\n(n={r['n']})" for g, r in zip(groups, rows)])
    ax.set_xlabel(y)
    ax.set_ylabel(x)
    ax.spines[["top", "right"]].set_visible(False)
    return VizResult(fig, ax, pd.DataFrame(rows))
