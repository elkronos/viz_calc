"""Correlation analysis: matrices, group differences and quadrant views."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from . import stats as st
from ._core import (
    NEUTRAL,
    OKABE_ITO,
    VizResult,
    category_order,
    check_choice,
    check_dataframe,
    check_numeric,
    cleanup_on_error,
    column_list,
    get_ax,
)

__all__ = ["correlogram", "compare_correlations", "quadrant_plot"]

Method = Literal["pearson", "spearman"]
Adjust = Literal["holm", "fdr_bh", "bonferroni", "none"]


def _numeric_columns(data: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    if columns is None:
        columns = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c]) and not pd.api.types.is_bool_dtype(data[c])]
    else:
        columns = column_list("columns", columns)
        check_dataframe(data, columns)
        check_numeric(data, *columns)
    if len(columns) < 2:
        raise ValueError("need at least two numeric columns")
    return list(columns)


def _pairwise(data: pd.DataFrame, cols: list[str], method: Method, level: float = 0.95) -> pd.DataFrame:
    rows = []
    for a, b in combinations(cols, 2):
        res = st.correlation_test(data[a], data[b], method, level)
        rows.append({"var1": a, "var2": b, **res})
    return pd.DataFrame(rows)


def _draw_matrix(ax: Axes, mat: np.ndarray, labels: list[str], triangle: str, vlim: float, cmap: str,
                 text: np.ndarray | None, colorbar_label: str) -> None:
    """Draw a symmetric matrix; triangles drop the empty first row / last column."""
    k = len(labels)
    rows, cols = np.arange(k), np.arange(k)
    mask = np.zeros((k, k), bool)
    if triangle == "lower":
        rows, cols = rows[1:], cols[:-1]
        mask = np.triu(np.ones((k, k), bool))
    elif triangle == "upper":
        rows, cols = rows[:-1], cols[1:]
        mask = np.tril(np.ones((k, k), bool))
    sub = np.ma.array(mat, mask=mask)[np.ix_(rows, cols)]
    im = ax.imshow(sub, cmap=cmap, vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(cols)), [labels[c] for c in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), [labels[r] for r in rows])
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if text is not None:
        for a, i in enumerate(rows):
            for b, j in enumerate(cols):
                if not mask[i, j] and text[i, j]:
                    ax.text(b, a, text[i, j], ha="center", va="center", fontsize=9,
                            color="white" if abs(mat[i, j]) > 0.6 * vlim else "black")
    ax.figure.colorbar(im, ax=ax, shrink=0.8, label=colorbar_label)


@cleanup_on_error
def correlogram(
    data: pd.DataFrame,
    columns: Sequence[str] | None = None,
    method: Method = "pearson",
    p_adjust: Adjust = "holm",
    alpha: float = 0.05,
    triangle: Literal["lower", "upper", "full"] = "lower",
    annotate: bool = True,
    hide_nonsignificant: bool = False,
    decimals: int = 2,
    cmap: str = "RdBu_r",
    ax: Axes | None = None,
) -> VizResult:
    """Correlation matrix heatmap with multiplicity-corrected significance.

    Testing every pair of *k* variables runs ``k(k-1)/2`` tests, so starring
    raw p-values below 0.05 produces false positives. Here p-values are
    adjusted across the unique pairs (Holm by default; ``"fdr_bh"`` suits
    exploratory screening). Missing values are dropped pairwise, and each
    pair's ``n`` is reported. Spearman p-values for pairs with at most 9
    complete observations are exact permutation p-values (see
    :func:`viz_calc.stats.correlation_test`), so a tiny pair cannot be starred
    on the strength of the large-sample approximation.

    A diverging, perceptually balanced colormap centred on zero is used so
    that sign and magnitude are both read correctly (Crameri et al., 2020).

    *columns* (any sequence of labels, e.g. ``df.columns[:4]``) defaults to
    every numeric, non-boolean column.

    Returns a long ``table`` with one row per pair: ``r``, Fisher-z CI,
    ``p``, ``p_adjusted``, ``n`` and ``significant``.

    References
    ----------
    Holm, S. (1979). *Scandinavian Journal of Statistics*, 6(2), 65–70.
    Benjamini, Y., & Hochberg, Y. (1995). *JRSS B*, 57(1), 289–300.
    Crameri, F., Shephard, G. E., & Heron, P. J. (2020). The misuse of colour
    in science communication. *Nature Communications*, 11, 5444.
    """
    check_dataframe(data)
    check_choice("method", method, ["pearson", "spearman"])
    check_choice("p_adjust", p_adjust, ["holm", "fdr_bh", "bonferroni", "none"])
    check_choice("triangle", triangle, ["lower", "upper", "full"])
    cols = _numeric_columns(data, columns)
    table = _pairwise(data, cols, method)
    table["p_adjusted"] = st.adjust_pvalues(table["p"], p_adjust)
    table["significant"] = table["p_adjusted"] < alpha

    k = len(cols)
    idx = {c: i for i, c in enumerate(cols)}
    mat = np.eye(k)
    sig = np.zeros((k, k), bool)
    for row in table.itertuples():
        i, j = idx[row.var1], idx[row.var2]
        mat[i, j] = mat[j, i] = row.r
        sig[i, j] = sig[j, i] = row.significant
    text = None
    if annotate:
        text = np.empty((k, k), dtype=object)
        for i in range(k):
            for j in range(k):
                if np.isnan(mat[i, j]) or (hide_nonsignificant and not sig[i, j]):
                    text[i, j] = ""
                else:
                    text[i, j] = f"{mat[i, j]:.{decimals}f}" + ("*" if sig[i, j] else "")
    full = mat.copy()
    if hide_nonsignificant:  # blank the drawn cells only; info["matrix"] keeps every coefficient
        mat = np.where(sig | np.eye(k, dtype=bool), mat, 0.0)

    fig, ax = get_ax(ax, figsize=(1.0 * k + 3, 0.9 * k + 2))
    _draw_matrix(ax, mat, cols, triangle, 1.0, cmap, text, f"{method.title()} correlation")
    note = "none" if p_adjust == "none" else p_adjust
    ax.set_title(f"{method.title()} correlations  (* p < {alpha}, {note}-adjusted)", loc="left", fontsize="medium")
    matrix = pd.DataFrame(np.where(np.eye(k, dtype=bool), 1.0, full), index=cols, columns=cols)
    return VizResult(fig, ax, table, {"matrix": matrix, "method": method, "p_adjust": p_adjust, "alpha": alpha})


@cleanup_on_error
def compare_correlations(
    data: pd.DataFrame,
    group: str,
    columns: Sequence[str] | None = None,
    method: Method = "pearson",
    p_adjust: Adjust = "holm",
    alpha: float = 0.05,
    order: Sequence[Any] | None = None,
    decimals: int = 2,
    cmap: str = "PuOr_r",
) -> VizResult:
    """Do correlations differ between groups? Fisher r-to-z tests, not eyeballing.

    For every pair of groups, draws the matrix of *differences*
    ``r(group B) − r(group A)`` and stars cells where Fisher's z-test for two
    independent correlations is significant after adjusting across the
    variable pairs in that comparison. Spearman correlations use the
    ``1.06/(n−3)`` variance of Fieller, Hartley & Pearson (1957).
    *columns* (any sequence of labels) defaults to every numeric,
    non-boolean column other than *group*.

    The groups must contain different units (independent samples). The
    z-test is asymptotic, so read it with caution when a group has only a few
    rows.

    References
    ----------
    Fisher, R. A. (1921). *Metron*, 1, 3–32.
    Cohen, J., Cohen, P., West, S. G., & Aiken, L. S. (2003). *Applied Multiple
    Regression/Correlation Analysis for the Behavioral Sciences* (3rd ed.).
    """
    check_dataframe(data, [group])
    check_choice("method", method, ["pearson", "spearman"])
    check_choice("p_adjust", p_adjust, ["holm", "fdr_bh", "bonferroni", "none"])
    cols = _numeric_columns(data.drop(columns=[group]), columns)
    groups = category_order(data[group], order)
    if len(groups) < 2:
        raise ValueError("need at least two groups")
    per_group = {g: _pairwise(data[data[group] == g], cols, method) for g in groups}

    frames = []
    for ga, gb in combinations(groups, 2):
        a, b = per_group[ga], per_group[gb]
        z, p = st.compare_correlations_test(b["r"], b["n"], a["r"], a["n"], method)
        f = pd.DataFrame({"group_a": ga, "group_b": gb, "var1": a["var1"], "var2": a["var2"],
                          "r_a": a["r"], "n_a": a["n"], "r_b": b["r"], "n_b": b["n"],
                          "difference": b["r"] - a["r"], "z": z, "p": p})
        f["p_adjusted"] = st.adjust_pvalues(f["p"], p_adjust)
        f["significant"] = f["p_adjusted"] < alpha
        frames.append(f)
    table = pd.concat(frames, ignore_index=True)

    pairs = list(combinations(groups, 2))
    k = len(cols)
    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * (0.9 * k + 3), nrows * (0.9 * k + 2)), squeeze=False)
    vlim = max(float(np.nanmax(np.abs(table["difference"]))), 0.1)
    idx = {c: i for i, c in enumerate(cols)}
    for ax, (ga, gb) in zip(axes.flat, pairs):
        sub = table[(table.group_a == ga) & (table.group_b == gb)]
        mat = np.zeros((k, k))
        text = np.full((k, k), "", dtype=object)
        for row in sub.itertuples():
            i, j = idx[row.var1], idx[row.var2]
            mat[i, j] = mat[j, i] = row.difference
            text[i, j] = text[j, i] = "" if np.isnan(row.difference) else f"{row.difference:+.{decimals}f}" + ("*" if row.significant else "")
        _draw_matrix(ax, mat, cols, "lower", vlim, cmap, text, "Δr")
        ax.set_title(f"r({gb}) − r({ga})", loc="left", fontsize="medium")
    for ax in list(axes.flat)[len(pairs):]:
        ax.set_visible(False)
    fig.suptitle(f"Differences in {method} correlations  (* p < {alpha}, Fisher z, {p_adjust}-adjusted)", x=0.01, ha="left")
    fig.tight_layout()
    return VizResult(fig, axes, table, {"method": method, "p_adjust": p_adjust, "alpha": alpha,
                                        "group_correlations": per_group})


@cleanup_on_error
def quadrant_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    center: Literal["mean", "median"] = "mean",
    standardize: bool = True,
    method: Method = "pearson",
    fit: bool = True,
    annotate: bool = True,
    label: str | None = None,
    color: str = OKABE_ITO[0],
    ax: Axes | None = None,
) -> VizResult:
    """Scatter plot split into quadrants around the centre of each variable.

    With ``standardize=True`` both axes are z-scores, so the quadrants read as
    "above/below average on x and y". Each quadrant is labelled with its share
    of points, and the correlation is reported with a Fisher-z CI.

    Parameters
    ----------
    center
        Split at the ``"mean"`` or ``"median"``. The median gives quadrants
        that are balanced on each axis when the data are skewed. Points exactly
        at the centre count as "high".
    label
        Optional column whose values annotate each point.
    fit
        Draw the ordinary least-squares line.
    """
    check_dataframe(data, [x, y, label])
    check_numeric(data, x, y)
    check_choice("center", center, ["mean", "median"])
    check_choice("method", method, ["pearson", "spearman"])
    d = data[[x, y] + ([label] if label is not None else [])].dropna(subset=[x, y])
    if len(d) < 3:
        raise ValueError("need at least three complete observations")
    constant = [c for c in (x, y) if d[c].nunique() < 2]
    if constant:
        raise ValueError(f"column(s) have a single value, so there are no quadrants: {constant}")
    xr, yr = d[x].to_numpy(float), d[y].to_numpy(float)
    # math.fsum gives the correctly rounded mean, so a point exactly on the mean is not pushed below it.
    rx = math.fsum(xr) / xr.size if center == "mean" else float(np.median(xr))
    ry = math.fsum(yr) / yr.size if center == "mean" else float(np.median(yr))
    # Classify on the raw values: standardizing cannot change which side a point is on, but its rounding
    # error can move a point that sits exactly on the mean across the line.
    eps = 4 * np.finfo(float).eps
    right = (xr >= rx) | np.isclose(xr, rx, rtol=eps, atol=0)  # points at the centre count as "high"
    top = (yr >= ry) | np.isclose(yr, ry, rtol=eps, atol=0)
    for name, side in ((x, right), (y, top)):
        if side.all() or not side.any():
            hint = " (heavy ties); try center='mean'" if center == "median" else ""
            raise ValueError(f"every point of {name!r} is on one side of its {center}{hint}")
    if standardize:
        xv, yv = (xr - xr.mean()) / xr.std(ddof=1), (yr - yr.mean()) / yr.std(ddof=1)
        cx, cy = (rx - xr.mean()) / xr.std(ddof=1), (ry - yr.mean()) / yr.std(ddof=1)
    else:
        xv, yv, cx, cy = xr, yr, rx, ry

    fig, ax = get_ax(ax, figsize=(7, 6))
    ax.scatter(xv, yv, s=22, color=color, alpha=0.7, lw=0)
    ax.axvline(cx, color="black", lw=0.8)
    ax.axhline(cy, color="black", lw=0.8)
    if fit and np.ptp(xv) > 0:
        slope, intercept = np.polyfit(xv, yv, 1)
        grid = np.linspace(xv.min(), xv.max(), 50)
        ax.plot(grid, intercept + slope * grid, color=OKABE_ITO[5], lw=1.5)
    if label is not None:
        for xi, yi, t in zip(xv, yv, d[label]):
            ax.annotate(str(t), (xi, yi), xytext=(3, 3), textcoords="offset points", fontsize=7, color=NEUTRAL)

    spec = [("high x, high y", right & top, 0.97, 0.97, "right", "top"),
            ("low x, high y", ~right & top, 0.03, 0.97, "left", "top"),
            ("low x, low y", ~right & ~top, 0.03, 0.03, "left", "bottom"),
            ("high x, low y", right & ~top, 0.97, 0.03, "right", "bottom")]
    rows = [{"quadrant": name, "n": int(m.sum()), "percent": 100 * m.mean()} for name, m, *_ in spec]
    if annotate:
        for (name, m, tx, ty, ha, va) in spec:
            ax.text(tx, ty, f"{name}\n{100 * m.mean():.1f}%", transform=ax.transAxes, ha=ha, va=va, fontsize=9,
                    color=NEUTRAL, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})

    cor = st.correlation_test(d[x], d[y], method)
    sym = "r" if method == "pearson" else "ρ"
    ax.set_title(f"{sym} = {cor['r']:.2f}  [95% CI {cor['ci_low']:.2f}, {cor['ci_high']:.2f}],  n = {cor['n']}",
                 loc="left", fontsize="medium")
    ax.set_xlabel(f"{x} (z-score)" if standardize else x)
    ax.set_ylabel(f"{y} (z-score)" if standardize else y)
    ax.spines[["top", "right"]].set_visible(False)
    return VizResult(fig, ax, pd.DataFrame(rows), {"correlation": cor, "center": (cx, cy), "standardized": standardize})
