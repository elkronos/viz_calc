"""Compare groups: estimation plots, benchmarks, rankings and survey scales."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from . import stats as st
from ._core import (
    NEUTRAL,
    OKABE_ITO,
    VizResult,
    abbreviate,
    category_order,
    check_choice,
    check_dataframe,
    check_numeric,
    get_ax,
    palette,
    text_color,
)

__all__ = ["estimation_plot", "benchmark_bar", "lollipop", "dumbbell", "divergent_bar", "centered_bar", "likert"]


def _jitter(n: int, width: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-width, width, n) if n > 1 else np.zeros(n)


def estimation_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    reference: Any = None,
    order: Sequence[Any] | None = None,
    level: float = 0.95,
    n_resamples: int = 5000,
    p_adjust: Literal["holm", "fdr_bh", "bonferroni", "none"] = "holm",
    colors: Sequence[str] | str | None = None,
    seed: int = 0,
    figsize: tuple[float, float] = (8, 7),
) -> VizResult:
    """Show every observation and the effect size, not just a p-value.

    A Cumming-style estimation plot (Ho et al., 2019). The upper panel shows
    the raw data per group with each group's mean and t-based CI. The lower
    panel shows the difference of each group's mean from the *reference*
    group: the bootstrap sampling distribution, the BCa confidence interval,
    and a zero line. Showing the data instead of "dynamite" bars follows
    Weissgerber et al. (2015).

    The returned table reports, for every comparison, the mean difference with
    its bootstrap CI, Hedges' *g* with CI, and a Welch *t*-test p-value
    adjusted for the number of comparisons (Holm by default).

    Parameters
    ----------
    data
        Long-format data: one row per observation.
    x
        Grouping column.
    y
        Numeric outcome column.
    reference
        Group the others are compared with. Defaults to the first group in
        *order*.
    order
        Order of the groups on the axis.
    level
        Confidence level for every interval.
    n_resamples
        Bootstrap resamples for the difference CI.
    p_adjust
        Multiple-comparison correction for the Welch p-values.
    colors
        One colour per group (defaults to the Okabe–Ito palette).
    seed
        Seed for jitter and bootstrap, so the figure is reproducible.

    Returns
    -------
    VizResult
        ``table`` has one row per group (``n``, ``mean``, ``sd``, CI) and
        ``info["comparisons"]`` has one row per comparison.

    References
    ----------
    Ho, J. et al. (2019). Moving beyond P values: data analysis with estimation
    graphics. *Nature Methods*, 16, 565–566.
    Cumming, G. (2014). The new statistics: why and how. *Psychological
    Science*, 25(1), 7–29.
    Weissgerber, T. L. et al. (2015). Beyond bar and line graphs: time for a
    new data presentation paradigm. *PLoS Biology*, 13(4), e1002128.
    """
    check_dataframe(data, [x, y])
    check_numeric(data, y)
    groups = category_order(data[x], order)
    if len(groups) < 2:
        raise ValueError("estimation_plot needs at least two groups")
    reference = groups[0] if reference is None else reference
    if reference not in groups:
        raise ValueError(f"reference {reference!r} is not one of the groups {groups}")
    samples = {g: data.loc[data[x] == g, y].dropna().to_numpy(float) for g in groups}
    small = [g for g, s in samples.items() if s.size < 2]
    if small:
        raise ValueError(f"groups need at least two observations: {small}")

    rng = np.random.default_rng(seed)
    cols = palette(len(groups), colors)
    fig, (ax_raw, ax_diff) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 2]})

    rows = []
    for i, g in enumerate(groups):
        s = samples[g]
        ax_raw.scatter(i - 0.08 + _jitter(s.size, 0.12, rng), s, s=14, color=cols[i], alpha=0.6, linewidths=0)
        m, lo, hi = st.mean_ci(s, level)
        ax_raw.errorbar(i + 0.22, m, yerr=[[m - lo], [hi - m]], fmt="o", color="black", ms=5, capsize=0, lw=1.5)
        rows.append({x: g, "n": s.size, "mean": m, "sd": s.std(ddof=1), "ci_low": lo, "ci_high": hi})
    ax_raw.set_ylabel(y)
    ax_raw.set_title(f"{y} by {x}: raw data with mean and {level:.0%} CI", loc="left")

    comps = []
    ref = samples[reference]
    ax_diff.axhline(0, color=NEUTRAL, lw=1, ls="--")
    for i, g in enumerate(groups):
        if g == reference:
            ax_diff.plot(i, 0, marker="_", color="black", ms=14)
            continue
        est, lo, hi, dist = st._bootstrap((ref, samples[g]), lambda a, b: np.mean(b) - np.mean(a),
                                          level, n_resamples, "BCa", seed)
        dens = _half_violin(dist)
        if dens is not None:
            grid, width = dens
            ax_diff.fill_betweenx(grid, i, i + width * 0.35, color=cols[i], alpha=0.35, lw=0)
        ax_diff.errorbar(i, est, yerr=[[est - lo], [hi - est]], fmt="o", color="black", ms=6, lw=2)
        w = st.welch_test(ref, samples[g], level)
        h = st.hedges_g(ref, samples[g], level)
        comps.append({"group": g, "reference": reference, "difference": est, "ci_low": lo, "ci_high": hi,
                      "hedges_g": h["g"], "g_ci_low": h["ci_low"], "g_ci_high": h["ci_high"],
                      "welch_t": w["t"], "df": w["df"], "p": w["p"]})
    comps_df = pd.DataFrame(comps)
    comps_df["p_adjusted"] = st.adjust_pvalues(comps_df["p"], p_adjust)
    ax_diff.set_ylabel(f"Difference from\n{reference}")
    ax_diff.set_xticks(range(len(groups)), [str(g) for g in groups])
    ax_diff.set_xlabel(x)
    ax_diff.set_title(f"Mean difference with bootstrap {level:.0%} CI (BCa)", loc="left", fontsize="medium")
    for ax in (ax_raw, ax_diff):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return VizResult(fig, np.array([ax_raw, ax_diff]), pd.DataFrame(rows),
                     {"comparisons": comps_df, "level": level, "p_adjust": p_adjust, "reference": reference})


def _half_violin(values: np.ndarray, points: int = 200):
    from scipy.stats import gaussian_kde

    values = values[np.isfinite(values)]
    if values.size < 3 or np.ptp(values) == 0:
        return None
    kde = gaussian_kde(values)
    grid = np.linspace(values.min(), values.max(), points)
    dens = kde(grid)
    return grid, dens / dens.max()


def benchmark_bar(
    data: pd.DataFrame,
    x: str,
    y: str,
    threshold: float | Literal["mean", "median"] = "mean",
    error: Literal["ci", "se", "sd"] = "ci",
    classify: Literal["ci", "mean"] = "ci",
    level: float = 0.95,
    order: Sequence[Any] | None = None,
    sort: bool = False,
    show_points: bool = True,
    colors: tuple[str, str, str] = (OKABE_ITO[5], OKABE_ITO[0], "#bdbdbd"),
    ax: Axes | None = None,
) -> VizResult:
    """Group means against a benchmark, flagging groups that clearly differ.

    With ``classify="ci"`` (default) a group is coloured *above* or *below*
    only when its confidence interval excludes the benchmark; groups whose CI
    contains it are grey (*indistinguishable*). This avoids reading noise as a
    difference, which colouring by the point estimate alone invites.

    Error bars default to a t-based CI rather than ±1 SE, because a CI has a
    stated coverage and SE bars are often misread as one (Cumming & Finch,
    2005). Individual points are overlaid by default (Weissgerber et al.,
    2015).

    Parameters
    ----------
    threshold
        A number, or ``"mean"``/``"median"`` of all observations of *y*.
    error
        What the error bars show: ``"ci"``, ``"se"`` or ``"sd"``.
    classify
        ``"ci"`` colours by whether the CI excludes the threshold; ``"mean"``
        by which side the mean falls on.
    colors
        ``(below, above, indistinguishable)`` colours.

    References
    ----------
    Cumming, G., & Finch, S. (2005). Inference by eye: confidence intervals and
    how to read pictures of data. *American Psychologist*, 60(2), 170–180.
    """
    check_dataframe(data, [x, y])
    check_numeric(data, y)
    check_choice("error", error, ["ci", "se", "sd"])
    check_choice("classify", classify, ["ci", "mean"])
    if threshold == "mean":
        thr = float(data[y].mean())
    elif threshold == "median":
        thr = float(data[y].median())
    elif isinstance(threshold, (int, float, np.number)) and not isinstance(threshold, bool):
        thr = float(threshold)
    else:
        raise ValueError("threshold must be a number, 'mean' or 'median'")

    groups = category_order(data[x], order)
    rows = []
    for g in groups:
        s = data.loc[data[x] == g, y].dropna().to_numpy(float)
        m, lo, hi = st.mean_ci(s, level)
        sd = s.std(ddof=1) if s.size > 1 else np.nan
        if classify == "ci":
            status = "above" if lo > thr else "below" if hi < thr else "indistinguishable"
        else:
            status = "above" if m >= thr else "below"
        rows.append({x: g, "n": s.size, "mean": m, "sd": sd, "se": sd / np.sqrt(s.size) if s.size else np.nan,
                     "ci_low": lo, "ci_high": hi, "status": status})
    table = pd.DataFrame(rows)
    if sort:
        table = table.sort_values("mean", ascending=False, ignore_index=True)

    if error == "ci":
        err = np.vstack([table["mean"] - table["ci_low"], table["ci_high"] - table["mean"]])
        err_text = f"{level:.0%} CI"
    else:
        err = table[error].to_numpy()
        err_text = "±1 SE" if error == "se" else "±1 SD"

    color_for = {"below": colors[0], "above": colors[1], "indistinguishable": colors[2]}
    fig, ax = get_ax(ax)
    pos = np.arange(len(table))
    ax.bar(pos, table["mean"], color=[color_for[s] for s in table["status"]], width=0.65, zorder=2)
    ax.errorbar(pos, table["mean"], yerr=err, fmt="none", ecolor="black", capsize=4, lw=1.2, zorder=3)
    if show_points:
        rng = np.random.default_rng(0)
        for p, g in zip(pos, table[x]):
            s = data.loc[data[x] == g, y].dropna().to_numpy(float)
            ax.scatter(p + _jitter(s.size, 0.18, rng), s, s=8, color="black", alpha=0.35, lw=0, zorder=4)
    ax.axhline(thr, color="black", ls="--", lw=1, zorder=5)
    ax.set_xticks(pos, [str(g) for g in table[x]])
    ax.set_xlabel(x)
    ax.set_ylabel(f"Mean {y}")
    ax.set_title(f"Mean {y} by {x} vs benchmark {thr:.3g} (bars: {err_text})", loc="left")
    present = [s for s in ("above", "below", "indistinguishable") if s in set(table["status"])]
    ax.legend(handles=[Patch(color=color_for[s], label=s) for s in present], frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1))
    ax.spines[["top", "right"]].set_visible(False)
    return VizResult(fig, ax, table, {"threshold": thr, "error": error, "classify": classify, "level": level})


def lollipop(
    data: pd.DataFrame,
    x: str,
    y: str | None = None,
    stat: Literal["mean", "median", "sum", "count"] = "mean",
    sort: Literal["descending", "ascending"] | None = "descending",
    threshold: float | None = None,
    labels: bool = True,
    horizontal: bool = True,
    color: str = OKABE_ITO[0],
    ax: Axes | None = None,
) -> VizResult:
    """Rank categories by a summary statistic as a lollipop (dot) chart.

    Dot plots are read more accurately than bars for comparing values on a
    common scale, and horizontal labels stay legible (Cleveland & McGill,
    1984). Stems start at zero so lengths stay honest. Values below
    *threshold* are faded.

    References
    ----------
    Cleveland, W. S., & McGill, R. (1984). Graphical perception. *JASA*,
    79(387), 531–554.
    """
    check_dataframe(data, [x, y])
    check_choice("stat", stat, ["mean", "median", "sum", "count"])
    if stat != "count":
        if y is None:
            raise ValueError(f"stat={stat!r} needs a y column")
        check_numeric(data, y)
        agg = data.groupby(x, observed=True)[y].agg(stat)
    else:
        agg = data.groupby(x, observed=True).size()
    table = agg.rename("value").reset_index()
    if sort:
        table = table.sort_values("value", ascending=(sort == "ascending"), ignore_index=True)

    fig, ax = get_ax(ax, figsize=(8, max(3, 0.4 * len(table) + 1)) if horizontal else (8, 5))
    pos = np.arange(len(table))
    if horizontal:
        pos = pos[::-1]  # first row at the top
    alphas = np.where(table["value"] < threshold, 0.3, 1.0) if threshold is not None else np.ones(len(table))
    for p, v, a in zip(pos, table["value"], alphas):
        if horizontal:
            ax.hlines(p, 0, v, color=color, alpha=a, lw=1.5)
        else:
            ax.vlines(p, 0, v, color=color, alpha=a, lw=1.5)
    xs, ys = (table["value"], pos) if horizontal else (pos, table["value"])
    ax.scatter(xs, ys, s=90, color=color, alpha=alphas, zorder=3)
    if labels:
        for p, v in zip(pos, table["value"]):
            if horizontal:
                ax.annotate(abbreviate(v), (v, p), xytext=(8 if v >= 0 else -8, 0), textcoords="offset points",
                            va="center", ha="left" if v >= 0 else "right", fontsize=9)
            else:
                ax.annotate(abbreviate(v), (p, v), xytext=(0, 8 if v >= 0 else -8), textcoords="offset points",
                            ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ticks = [str(v) for v in table[x]]
    label = f"{stat} of {y}" if stat != "count" else "count"
    fmt = FuncFormatter(lambda v, _: abbreviate(v))
    if horizontal:
        ax.set_yticks(pos, ticks)
        ax.set_xlabel(label)
        ax.xaxis.set_major_formatter(fmt)
        ax.axvline(0, color="black", lw=0.8)
        if threshold is not None:
            ax.axvline(threshold, color=NEUTRAL, ls="--", lw=1)
        ax.margins(x=0.12)
    else:
        ax.set_xticks(pos, ticks, rotation=45, ha="right")
        ax.set_ylabel(label)
        ax.yaxis.set_major_formatter(fmt)
        ax.axhline(0, color="black", lw=0.8)
        if threshold is not None:
            ax.axhline(threshold, color=NEUTRAL, ls="--", lw=1)
        ax.margins(y=0.12)
    ax.spines[["top", "right"]].set_visible(False)
    return VizResult(fig, ax, table, {"stat": stat, "threshold": threshold})


def dumbbell(
    data: pd.DataFrame,
    label: str,
    start: str,
    end: str,
    sort: bool = True,
    show_values: bool = False,
    start_label: str | None = None,
    end_label: str | None = None,
    colors: tuple[str, str] = (OKABE_ITO[4], OKABE_ITO[0]),
    ax: Axes | None = None,
) -> VizResult:
    """Before/after (or A/B) values per item, connected to show the change.

    Rows are sorted by the change so the largest movers are easy to find. The
    table reports absolute and percentage change.
    """
    check_dataframe(data, [label, start, end])
    check_numeric(data, start, end)
    table = data[[label, start, end]].copy()
    table["change"] = table[end] - table[start]
    with np.errstate(divide="ignore", invalid="ignore"):
        table["pct_change"] = np.where(table[start] != 0, table["change"] / table[start].abs() * 100, np.nan)
    if sort:
        table = table.sort_values("change", ignore_index=True)
    else:
        table = table.reset_index(drop=True)

    fig, ax = get_ax(ax, figsize=(8, max(3, 0.45 * len(table) + 1)))
    pos = np.arange(len(table))
    ax.hlines(pos, table[start], table[end], color="#c7c7c7", lw=3, zorder=1)
    ax.scatter(table[start], pos, s=80, color=colors[0], label=start_label or start, zorder=2)
    ax.scatter(table[end], pos, s=80, color=colors[1], label=end_label or end, zorder=3)
    if show_values:
        for p, a, b in zip(pos, table[start], table[end]):
            left, right = (a, b) if a <= b else (b, a)
            ax.annotate(abbreviate(left), (left, p), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=8)
            ax.annotate(abbreviate(right), (right, p), xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=8)
        ax.margins(x=0.1)
    ax.set_yticks(pos, [str(v) for v in table[label]])
    ax.set_ylabel(label)
    ax.set_xlabel("Value")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: abbreviate(v)))
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)
    return VizResult(fig, ax, table)


def divergent_bar(
    data: pd.DataFrame,
    category: str,
    left: str,
    right: str,
    left_label: str | None = None,
    right_label: str | None = None,
    sort: bool = False,
    colors: tuple[str, str] = (OKABE_ITO[1], OKABE_ITO[0]),
    ax: Axes | None = None,
) -> VizResult:
    """Two measures per category drawn back to back (e.g. a population pyramid).

    *left* values are drawn to the left of zero, *right* values to the right.
    Axis labels show absolute values, so nothing reads as negative.
    Rows with a repeated category are summed.
    """
    check_dataframe(data, [category, left, right])
    check_numeric(data, left, right)
    if (data[[left, right]] < 0).any().any():
        raise ValueError("divergent_bar expects non-negative values in both columns")
    table = data.groupby(category, sort=False, observed=True)[[left, right]].sum().reset_index()
    if sort:
        table["total"] = table[left] + table[right]
        table = table.sort_values("total", ignore_index=True).drop(columns="total")
    fig, ax = get_ax(ax, figsize=(8, max(3, 0.4 * len(table) + 1)))
    pos = np.arange(len(table))
    ax.barh(pos, -table[left], color=colors[0], label=left_label or left)
    ax.barh(pos, table[right], color=colors[1], label=right_label or right)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(pos, [str(v) for v in table[category]])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: abbreviate(abs(v))))
    lim = max(table[left].max(), table[right].max()) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylabel(category)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    return VizResult(fig, ax, table)


def centered_bar(
    data: pd.DataFrame,
    x: str,
    y: str,
    threshold: float | Literal["mean", "median"] = "mean",
    level: float = 0.95,
    order: Sequence[Any] | None = None,
    labels: bool = True,
    colors: tuple[str, str] = (OKABE_ITO[0], OKABE_ITO[1]),
    ax: Axes | None = None,
) -> VizResult:
    """Share of each group at-or-above vs below a threshold, centred on zero.

    Error bars are Wilson score intervals for the at-or-above proportion
    (Wilson, 1927), which behave well for small groups and extreme
    proportions (Brown, Cai & DasGupta, 2001).
    """
    check_dataframe(data, [x, y])
    check_numeric(data, y)
    thr = float(data[y].mean()) if threshold == "mean" else float(data[y].median()) if threshold == "median" else float(threshold)
    groups = category_order(data[x], order)
    rows = []
    for g in groups:
        s = data.loc[data[x] == g, y].dropna()
        n, k = int(s.size), int((s >= thr).sum())
        lo, hi = st.wilson_ci(k, n, level)
        rows.append({x: g, "n": n, "n_above": k, "p_above": k / n if n else np.nan,
                     "ci_low": float(lo), "ci_high": float(hi), "p_below": (n - k) / n if n else np.nan})
    table = pd.DataFrame(rows)

    fig, ax = get_ax(ax)
    pos = np.arange(len(table))
    up = ax.bar(pos, table["p_above"], color=colors[0], label=f"≥ {thr:.3g}")
    down = ax.bar(pos, -table["p_below"], color=colors[1], label=f"< {thr:.3g}")
    ax.errorbar(pos, table["p_above"], yerr=[table["p_above"] - table["ci_low"], table["ci_high"] - table["p_above"]],
                fmt="none", ecolor="black", capsize=3, lw=1)
    if labels:
        ax.bar_label(up, labels=[f"{v:.0%}" for v in table["p_above"]], padding=2, fontsize=8, label_type="center", color="white")
        ax.bar_label(down, labels=[f"{v:.0%}" for v in table["p_below"]], padding=2, fontsize=8, label_type="center", color="black")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0%}"))
    ax.set_xticks(pos, [str(g) for g in table[x]])
    ax.set_xlabel(x)
    ax.set_ylabel(f"Share of {y}")
    ax.set_title(f"Share of {y} above/below {thr:.3g} (error bars: {level:.0%} Wilson CI)", loc="left")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax.spines[["top", "right"]].set_visible(False)
    return VizResult(fig, ax, table, {"threshold": thr, "level": level})


def likert(
    data: pd.DataFrame,
    items: Sequence[str],
    levels: Sequence[Any],
    sort: bool = True,
    colors: str | Sequence[str] = "RdBu",
    labels: bool = True,
    ax: Axes | None = None,
) -> VizResult:
    """Diverging stacked bars for Likert-type survey items.

    Each item is a bar of response shares, aligned so negative responses
    extend left of zero and positive ones right. An odd number of levels puts
    the neutral share straddling zero, as recommended by Robbins & Heiberger
    (2011). Items are sorted by *net agreement* (positive % minus negative %).

    Parameters
    ----------
    data
        Wide survey data: one row per respondent, one column per item.
    items
        Columns to plot, each holding responses from *levels*.
    levels
        Response options ordered from most negative to most positive.
    colors
        A diverging colormap name or one colour per level.

    References
    ----------
    Robbins, N. B., & Heiberger, R. M. (2011). Plotting Likert and other rating
    scales. *Proceedings of the 2011 Joint Statistical Meeting*, Section on
    Survey Research Methods, 1058–1066.
    """
    check_dataframe(data, items)
    levels = list(levels)
    if len(levels) < 2:
        raise ValueError("levels needs at least two response options")
    unknown = sorted({str(v) for c in items for v in data[c].dropna().unique() if v not in levels})
    if unknown:
        raise ValueError(f"responses not listed in levels: {unknown}")
    k = len(levels)
    half = k // 2
    neutral = levels[half] if k % 2 else None
    neg, pos_levels = levels[:half], levels[k - half:]

    rows = []
    for item in items:
        s = data[item].dropna()
        shares = s.value_counts(normalize=True).reindex(levels, fill_value=0.0) * 100
        rows.append({"item": item, "n": int(s.size), **shares.to_dict(),
                     "net": shares[pos_levels].sum() - shares[neg].sum()})
    table = pd.DataFrame(rows)
    if sort:
        table = table.sort_values("net", ignore_index=True)

    cols = palette(k, colors)
    fig, ax = get_ax(ax, figsize=(9, max(3, 0.5 * len(table) + 1.5)))
    ypos = np.arange(len(table))
    start = -(table[neg].sum(axis=1) + (table[neutral] / 2 if neutral is not None else 0))
    left = start.to_numpy(dtype=float).copy()
    for lvl, col in zip(levels, cols):
        width = table[lvl].to_numpy(float)
        bars = ax.barh(ypos, width, left=left, color=col, label=str(lvl), edgecolor="white", lw=0.5)
        if labels:
            ax.bar_label(bars, labels=[f"{w:.0f}" if w >= 6 else "" for w in width], label_type="center", fontsize=8,
                         color=text_color(col))
        left += width
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos, [f"{i} (n={n})" for i, n in zip(table["item"], table["n"])])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0f}%"))
    ax.set_xlim(-max(float((-start).max()), 1) * 1.05, max(float(left.max()), 1) * 1.05)
    ax.set_xlabel("Share of responses")
    ax.legend(frameon=False, ncol=k, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.spines[["top", "right", "left"]].set_visible(False)
    return VizResult(fig, ax, table, {"levels": levels, "neutral": neutral})
