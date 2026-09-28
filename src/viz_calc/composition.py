"""Part-to-whole charts: shares, flows between stages, and set overlaps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter

from . import stats as st
from ._core import (
    NEUTRAL,
    OKABE_ITO,
    VizResult,
    abbreviate,
    check_choice,
    check_dataframe,
    check_numeric,
    get_ax,
    palette,
    text_color,
)
from ._core import (
    category_order as _levels,
)

__all__ = ["waffle", "percent_grid", "stacked_percentages", "donut_grid", "nested_pie", "circular_bar",
           "waterfall", "funnel", "bullet", "upset"]


def _aggregate(data: pd.DataFrame, category: str, value: str | None) -> pd.Series:
    if value is None:
        return data[category].value_counts(sort=False)
    check_numeric(data, value)
    return data.groupby(category, sort=False, observed=True)[value].sum()


def waffle(
    data: pd.DataFrame,
    category: str,
    value: str | None = None,
    rows: int = 10,
    columns: int = 10,
    order: Sequence[Any] | None = None,
    colors: Sequence[str] | str | None = None,
    ax: Axes | None = None,
) -> VizResult:
    """Waffle chart: a grid of tiles apportioned by share.

    Tiles are allocated with the largest-remainder (Hamilton) method, so the
    grid is always exactly full; rounding each share independently can over-
    or under-fill it. With *value* omitted, rows are counted. Categories are
    drawn largest first unless *order* is given; *order* must list every
    category.
    """
    check_dataframe(data, [category, value])
    totals = _aggregate(data, category, value)
    if (totals < 0).any():
        raise ValueError("waffle values must be non-negative")
    if totals.sum() == 0:
        raise ValueError("waffle needs a positive total; every value is zero")
    if order is not None:  # must list every category, or the shares would be renormalized over a subset
        levels = _levels(data[category], order, complete=True)
    else:
        levels = list(totals.sort_values(ascending=False, kind="stable").index)
    totals = totals.reindex(levels).fillna(0)
    n_tiles = rows * columns
    tiles = st.largest_remainder(totals.to_numpy(float), n_tiles)
    cols = palette(len(levels), colors)

    fig, ax = get_ax(ax, figsize=(columns * 0.45 + 3, rows * 0.45 + 1))
    idx = 0
    for c, t in enumerate(tiles):
        for _ in range(t):
            r, k = divmod(idx, columns)
            ax.add_patch(plt.Rectangle((k, rows - 1 - r), 0.9, 0.9, color=cols[c]))
            idx += 1
    ax.set_xlim(-0.1, columns)
    ax.set_ylim(-0.1, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    share = totals / totals.sum() * 100 if totals.sum() else totals * 0
    ax.legend(handles=[Patch(color=cols[i], label=f"{lvl}: {abbreviate(totals[lvl])} ({share[lvl]:.1f}%)")
                       for i, lvl in enumerate(levels)],
              frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title(f"1 tile ≈ {100 / n_tiles:.2g}% of total", loc="left", fontsize="small", color=NEUTRAL)
    table = pd.DataFrame({category: levels, "value": totals.to_numpy(), "percent": share.to_numpy(), "tiles": tiles})
    return VizResult(fig, ax, table)


def percent_grid(
    data: pd.DataFrame,
    column: str,
    success: Any = None,
    facet: str | None = None,
    facet_order: Sequence[Any] | None = None,
    level: float = 0.95,
    colors: tuple[str, str] = (OKABE_ITO[0], "#dddddd"),
    col_wrap: int = 4,
) -> VizResult:
    """A 10×10 dot grid per group showing the percentage of "successes".

    Each title reports the percentage, its Wilson score CI and *n*, so small
    groups are not over-read.

    Parameters
    ----------
    column
        A binary column (two distinct values, e.g. 0/1, True/False, "yes"/"no").
    success
        The value counted as success. Defaults to ``True``/``1`` for boolean or
        0/1 data; required otherwise.
    """
    check_dataframe(data, [column, facet])
    values = data[column].dropna()
    uniq = set(pd.unique(values))
    if len(uniq) > 2:
        raise ValueError(f"{column!r} must have at most two distinct values, found {len(uniq)}")
    if success is None:
        if uniq <= {0, 1}:
            success = 1
        else:
            raise ValueError(f"set success= to the value of {column!r} to count (found {sorted(map(str, uniq))})")
    elif len(uniq) == 2 and success not in uniq:
        raise ValueError(f"success={success!r} does not occur in {column!r}; its values are {sorted(map(str, uniq))}")
    facets = _levels(data[facet], facet_order) if facet else [None]
    ncol = min(col_wrap, len(facets))
    nrow = int(np.ceil(len(facets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.6 * nrow), squeeze=False)
    rows = []
    for ax, f in zip(axes.flat, facets):
        s = (data if f is None else data[data[facet] == f])[column].dropna()
        n, k = int(s.size), int((s == success).sum())
        p = k / n if n else np.nan
        lo, hi = st.wilson_ci(k, n, level)
        filled = int(round(p * 100)) if n else 0
        xs, ys = np.meshgrid(np.arange(10), np.arange(10)[::-1])
        fill = np.arange(100) < filled
        ax.scatter(xs.ravel(), ys.ravel(), s=70, c=np.where(fill, colors[0], colors[1]))
        ax.set_xlim(-0.7, 9.7)
        ax.set_ylim(-0.7, 9.7)
        ax.set_aspect("equal")
        ax.axis("off")
        head = f"{f}: " if f is not None else ""
        ax.set_title(f"{head}{p:.1%}\n{level:.0%} CI {float(lo):.1%}–{float(hi):.1%}, n={n}", fontsize="medium")
        rows.append({"facet": f, "n": n, "successes": k, "percent": 100 * p,
                     "ci_low": 100 * float(lo), "ci_high": 100 * float(hi)})
    for ax in list(axes.flat)[len(facets):]:
        ax.set_visible(False)
    fig.tight_layout()
    table = pd.DataFrame(rows)
    if not facet:
        table = table.drop(columns="facet")
    return VizResult(fig, axes, table, {"success": success, "level": level})


def stacked_percentages(
    data: pd.DataFrame,
    group: str,
    category: str,
    group_order: Sequence[Any] | None = None,
    category_order: Sequence[Any] | None = None,
    horizontal: bool = False,
    min_label: float = 5.0,
    colors: Sequence[str] | str | None = None,
    ax: Axes | None = None,
) -> VizResult:
    """100 % stacked bars: the share of each *category* within each *group*.

    Group sizes (*n*) are shown in the tick labels because percentages hide
    them. Segments smaller than *min_label* percent are not labelled. Rows
    with a missing *category* are left out of *n* and the percentages.

    *group_order* may list a subset of groups. *category_order* sets the
    stacking order and must list every category, so shares always add to
    100 %.
    """
    check_dataframe(data, [group, category])
    groups = _levels(data[group], group_order)
    cats = _levels(data[category], category_order, complete=True)
    counts = pd.crosstab(data[group], data[category]).reindex(index=groups, columns=cats, fill_value=0)
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    cols = palette(len(cats), colors)
    fig, ax = get_ax(ax, figsize=(9, max(3, 0.5 * len(groups) + 1.5)) if horizontal else (max(6, 0.9 * len(groups) + 3), 5))
    pos = np.arange(len(groups))
    base = np.zeros(len(groups))
    for c, col in zip(cats, cols):
        vals = pct[c].to_numpy()
        bars = (ax.barh(pos, vals, left=base, color=col, label=str(c), edgecolor="white")
                if horizontal else ax.bar(pos, vals, bottom=base, color=col, label=str(c), edgecolor="white"))
        ax.bar_label(bars, labels=[f"{v:.0f}%" if v >= min_label else "" for v in vals], label_type="center",
                     fontsize=8, color=text_color(col))
        base += vals
    ticks = [f"{g}\n(n={n})" for g, n in zip(groups, counts.sum(axis=1))]
    if horizontal:
        ax.set_yticks(pos, ticks)
        ax.xaxis.set_major_formatter(PercentFormatter())
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
    else:
        ax.set_xticks(pos, ticks)
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_ylim(0, 100)
    ax.legend(title=category, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.spines[["top", "right"]].set_visible(False)
    table = counts.stack().rename("n").reset_index()
    group_n = table.groupby(group)["n"].transform("sum")
    table["percent"] = (table["n"] / group_n.replace(0, np.nan) * 100).fillna(0)
    return VizResult(fig, ax, table)


def donut_grid(
    data: pd.DataFrame,
    columns: Sequence[str] | None = None,
    labels: bool = True,
    min_label: float = 4.0,
    col_wrap: int = 3,
    colors: Sequence[str] | str | None = None,
) -> VizResult:
    """One donut per row of a wide table (rows = charts, columns = categories).

    Every category keeps the same colour in every donut. The row total is
    printed in the centre. Shares below *min_label* percent are not labelled.

    Note: people compare angles less accurately than lengths (Cleveland &
    McGill, 1984); for precise comparisons use :func:`stacked_percentages`.
    """
    check_dataframe(data, columns or [])
    columns = list(columns) if columns is not None else [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    check_numeric(data, *columns)
    if (data[columns] < 0).any().any():
        raise ValueError("donut values must be non-negative")
    cols = palette(len(columns), colors)
    n = len(data)
    ncol = min(col_wrap, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol + 1.5, 3.3 * nrow), squeeze=False)
    for ax, (idx, row) in zip(axes.flat, data[columns].iterrows()):
        vals = row.to_numpy(float)
        total = vals.sum()
        share = vals / total * 100 if total else vals
        wedges, _ = ax.pie(vals if total else np.ones_like(vals), colors=cols if total else ["#eeeeee"] * len(vals),
                           startangle=90, counterclock=False, wedgeprops={"width": 0.4, "edgecolor": "white"})
        if labels and total:
            for w, sh in zip(wedges, share):
                if sh >= min_label:
                    ang = np.deg2rad((w.theta1 + w.theta2) / 2)
                    ax.text(0.8 * np.cos(ang), 0.8 * np.sin(ang), f"{sh:.0f}%", ha="center", va="center", fontsize=8)
        ax.text(0, 0, abbreviate(total), ha="center", va="center", fontsize=11, weight="bold")
        ax.set_title(str(idx), fontsize="medium")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.legend(handles=[Patch(color=c, label=str(k)) for k, c in zip(columns, cols)], frameon=False,
               loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    table = data[columns].div(data[columns].sum(axis=1).replace(0, np.nan), axis=0).mul(100)
    return VizResult(fig, axes, table.rename_axis("row").reset_index().melt(id_vars="row", var_name="category",
                                                                             value_name="percent"))


def nested_pie(
    data: pd.DataFrame,
    outer: str,
    inner: str,
    value: str | None = None,
    labels: bool = True,
    colors: Sequence[str] | str | None = None,
    ax: Axes | None = None,
) -> VizResult:
    """Two-level donut: *outer* categories on the inner ring, *inner* sub-categories around them.

    Sub-category wedges use lighter shades of their parent's colour and sit
    directly outside it, so the hierarchy is visible. With *value* omitted,
    rows are counted.
    """
    check_dataframe(data, [outer, inner, value])
    if value is None:
        agg = data.groupby([outer, inner], sort=False, observed=True).size().rename("value")
    else:
        check_numeric(data, value)
        agg = data.groupby([outer, inner], sort=False, observed=True)[value].sum().rename("value")
    agg = agg.reset_index()
    parents = list(pd.unique(agg[outer]))
    parent_tot = agg.groupby(outer, sort=False)["value"].sum().reindex(parents)
    base_cols = palette(len(parents), colors)
    child_cols = []
    for p, c in zip(parents, base_cols):
        k = int((agg[outer] == p).sum())
        rgb = np.array(to_rgb(c))
        child_cols += [tuple(rgb + (1 - rgb) * t) for t in np.linspace(0.25, 0.65, k)]
    agg = pd.concat([agg[agg[outer] == p] for p in parents], ignore_index=True)

    fig, ax = get_ax(ax, figsize=(7, 7))
    wedges, texts = ax.pie(parent_tot, radius=0.7, colors=base_cols, labels=[str(p) for p in parents] if labels else None,
                           labeldistance=0.75, wedgeprops={"width": 0.35, "edgecolor": "white"}, startangle=90,
                           counterclock=False, textprops={"weight": "bold", "ha": "center", "fontsize": 9})
    for t, c in zip(texts, base_cols):
        t.set_color(text_color(c))
    ax.pie(agg["value"], radius=1.0, colors=child_cols, labels=agg[inner].astype(str) if labels else None,
           labeldistance=1.06, wedgeprops={"width": 0.3, "edgecolor": "white"}, startangle=90, counterclock=False,
           textprops={"fontsize": 8})
    ax.set_aspect("equal")
    total = agg["value"].sum()
    agg["percent_of_total"] = agg["value"] / total * 100 if total else np.nan
    agg["percent_of_parent"] = agg["value"] / agg[outer].map(parent_tot) * 100
    return VizResult(fig, ax, agg)


def circular_bar(
    data: pd.DataFrame,
    label: str,
    value: str,
    group: str | None = None,
    gap: int = 2,
    sort: bool = True,
    colors: Sequence[str] | str | None = None,
    inner_radius: float = 0.35,
    ax: Axes | None = None,
) -> VizResult:
    """Bars arranged around a circle, optionally clustered by *group*.

    Compact for many items, but radial bars are harder to compare than a
    straight bar chart because outer bars look longer; use for overview,
    not precise comparison. Values must be non-negative.
    """
    check_dataframe(data, [label, value, group])
    check_numeric(data, value)
    if (data[value] < 0).any():
        raise ValueError("circular_bar values must be non-negative")
    d = data[[label, value] + ([group] if group else [])].copy()
    groups = _levels(d[group]) if group else [None]
    parts, slot_idx, slot = [], [], 0
    for g in groups:
        sub = d if g is None else d[d[group] == g]
        if sort:
            sub = sub.sort_values(value, ascending=False)
        parts.append(sub)
        slot_idx += range(slot, slot + len(sub))
        slot += len(sub) + (gap if group else 0)  # empty slots separate the groups
    ordered = pd.concat(parts, ignore_index=True)
    n_slots = max(slot, 1)
    theta = np.linspace(0, 2 * np.pi, n_slots, endpoint=False)[slot_idx]
    vmax = float(ordered[value].max()) or 1.0
    heights = ordered[value].to_numpy(float) / vmax
    cols = palette(len(groups), colors)
    col_for = [cols[groups.index(g)] for g in ordered[group]] if group else [cols[0]] * len(ordered)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    else:
        fig = ax.figure
    width = 2 * np.pi / n_slots * 0.9
    ax.bar(theta, heights, width=width, bottom=inner_radius, color=col_for, edgecolor="white", lw=0.5)
    for t, h, name in zip(theta, heights, ordered[label]):
        # The axis runs clockwise from north, so the on-screen angle is 90° − θ.
        deg = (90 - np.rad2deg(t)) % 360
        flip = 90 < deg < 270  # keep labels on the left half upright
        ax.text(t, inner_radius + h + 0.03, str(name), rotation=deg + 180 if flip else deg,
                rotation_mode="anchor", ha="right" if flip else "left", va="center", fontsize=7)
    if group:
        for g, c in zip(groups, cols):
            th = theta[(ordered[group] == g).to_numpy()]
            ax.plot(np.linspace(th.min(), th.max(), 30), np.full(30, inner_radius - 0.05), color=c, lw=2)
            ax.text(th.mean(), inner_radius - 0.16, str(g), ha="center", va="center", fontsize=9, weight="bold")
    ax.set_ylim(0, inner_radius + 1.35)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.axis("off")
    ax.set_title(f"{value} (bar length relative to max = {abbreviate(vmax)})", fontsize="medium")
    return VizResult(fig, ax, ordered)


def waterfall(
    data: pd.DataFrame,
    label: str,
    value: str,
    values_are: Literal["changes", "levels"] = "changes",
    start_label: str | None = None,
    total_label: str | None = "Total",
    colors: tuple[str, str, str] = (OKABE_ITO[2], OKABE_ITO[5], NEUTRAL),
    ax: Axes | None = None,
) -> VizResult:
    """Waterfall chart: how a starting value becomes a final value.

    Parameters
    ----------
    values_are
        ``"changes"``: each row is an increment (the first row may be the
        starting value, see *start_label*). ``"levels"``: each row is a running
        level and increments are computed as differences.
    start_label
        Treat the first row as the starting total, drawn as a total bar.
    total_label
        Label for a final total bar; ``None`` omits it.
    colors
        ``(increase, decrease, total)``.
    """
    check_dataframe(data, [label, value])
    check_numeric(data, value)
    labels = data[label].astype(str).tolist()
    vals = data[value].to_numpy(float)
    if values_are == "levels":
        changes = np.r_[vals[0], np.diff(vals)]
        if start_label is None:
            start_label = labels[0]
    elif values_are == "changes":
        changes = vals.copy()
    else:
        raise ValueError("values_are must be 'changes' or 'levels'")
    kinds = ["total" if (i == 0 and start_label is not None) else "change" for i in range(len(changes))]
    ends = np.cumsum(changes)
    starts = np.r_[0, ends[:-1]]
    rows = [{"label": lab, "kind": k, "change": c, "start": 0.0 if k == "total" else s, "end": e}
            for lab, k, c, s, e in zip(labels, kinds, changes, starts, ends)]
    if total_label is not None:
        rows.append({"label": total_label, "kind": "total", "change": ends[-1], "start": 0.0, "end": ends[-1]})
    table = pd.DataFrame(rows)

    fig, ax = get_ax(ax, figsize=(max(6, 0.8 * len(table) + 2), 5))
    pos = np.arange(len(table))
    color = [colors[2] if k == "total" else colors[0] if c >= 0 else colors[1] for k, c in zip(table["kind"], table["change"])]
    bottoms = np.minimum(table["start"], table["end"])
    heights = (table["end"] - table["start"]).abs()
    bars = ax.bar(pos, heights, bottom=bottoms, color=color, width=0.65)
    ax.bar_label(bars, labels=[abbreviate(e) if k == "total" else ("+" if c >= 0 else "") + abbreviate(c)
                               for k, c, e in zip(table["kind"], table["change"], table["end"])], fontsize=8, padding=2)
    for i in range(len(table) - 1):
        ax.plot([i + 0.33, i + 1 - 0.33], [table["end"][i]] * 2, color=NEUTRAL, lw=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(pos, table["label"], rotation=30, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: abbreviate(v)))
    ax.set_ylabel(value)
    ax.legend(handles=[Patch(color=colors[0], label="increase"), Patch(color=colors[1], label="decrease"),
                       Patch(color=colors[2], label="total")], frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax.spines[["top", "right"]].set_visible(False)
    return VizResult(fig, ax, table)


def funnel(
    data: pd.DataFrame,
    stage: str,
    value: str,
    color: str = OKABE_ITO[0],
    ax: Axes | None = None,
) -> VizResult:
    """Conversion funnel with step-to-step and overall conversion rates.

    Stages are drawn in row order. The table reports ``pct_of_first`` and
    ``pct_of_previous`` (step conversion) and the ``drop_off`` count.
    """
    check_dataframe(data, [stage, value])
    check_numeric(data, value)
    t = data[[stage, value]].reset_index(drop=True).copy()
    v = t[value].to_numpy(float)
    t["pct_of_first"] = v / v[0] * 100 if len(v) and v[0] else np.nan
    t["pct_of_previous"] = np.r_[100.0, v[1:] / np.where(v[:-1] == 0, np.nan, v[:-1]) * 100] if len(v) else []
    t["drop_off"] = np.r_[0.0, v[:-1] - v[1:]] if len(v) else []
    fig, ax = get_ax(ax, figsize=(8, 0.7 * len(t) + 1.5))
    pos = np.arange(len(t))[::-1]
    ax.barh(pos, v, left=-v / 2, color=color, height=0.75)
    for p, row in zip(pos, t.itertuples(index=False)):
        vv, pf, pp = row[1], row[2], row[3]
        text = f"{abbreviate(vv)}  ({pf:.0f}%)"
        if vv >= 0.3 * v.max():
            ax.text(0, p, text, ha="center", va="center", color=text_color(color), weight="bold", fontsize=9)
        else:  # too narrow to hold the label: put it beside the bar
            ax.text(vv / 2 + v.max() * 0.02, p, text, ha="left", va="center", color="black", weight="bold", fontsize=9)
        if p != pos[0]:
            ax.text(v.max() / 2 * 1.02, p + 0.5, f"↓ {pp:.0f}% of previous", va="center", fontsize=8, color=NEUTRAL)
    ax.set_yticks(pos, t[stage].astype(str))
    ax.set_xlim(-v.max() / 2 * 1.05, v.max() / 2 * 1.6)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    return VizResult(fig, ax, t)


def bullet(
    data: pd.DataFrame,
    label: str,
    value: str,
    target: str | None = None,
    bands: Sequence[str] | Sequence[float] | None = None,
    bar_color: str = "#222222",
    ax: Axes | None = None,
) -> VizResult:
    """Bullet graphs: a measure against a target and qualitative ranges.

    Follows Stephen Few's specification: one dark bar for the measure, a
    short perpendicular line for the target, and background bands in shades
    of grey for qualitative ranges (e.g. poor/satisfactory/good), which stay
    readable in greyscale and for colour-blind readers.

    Parameters
    ----------
    bands
        Either column names holding each row's band upper limits, or a list of
        numeric limits shared by every row. Ascending order.

    References
    ----------
    Few, S. (2013). *Bullet Graph Design Specification*. Perceptual Edge.
    """
    band_cols = [b for b in (bands or []) if isinstance(b, str)]
    check_dataframe(data, [label, value, target, *band_cols])
    check_numeric(data, value, *([target] if target else []), *band_cols)
    n = len(data)
    fig, axes = plt.subplots(n, 1, figsize=(8, 0.9 * n + 0.6), squeeze=False)
    rows = []
    for ax, (_, row) in zip(axes[:, 0], data.iterrows()):
        limits = [float(row[b]) for b in band_cols] if band_cols else [float(b) for b in (bands or [])]
        top = max(limits + [float(row[value]), float(row[target]) if target else 0.0]) or 1.0
        greys = plt.get_cmap("Greys")(np.linspace(0.45, 0.15, max(len(limits), 1)))
        prev = 0.0
        for lim, g in zip(sorted(limits), greys):
            ax.barh(0, lim - prev, left=prev, height=1, color=g)
            prev = lim
        ax.barh(0, row[value], height=0.35, color=bar_color)
        if target:
            ax.plot([row[target]] * 2, [-0.35, 0.35], color="black", lw=2.5)
        ax.set_xlim(0, top * 1.02)
        ax.set_yticks([0], [str(row[label])])
        ax.set_ylim(-0.5, 0.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        rec = {label: row[label], "value": row[value]}
        if target:
            rec.update(target=row[target], pct_of_target=row[value] / row[target] * 100 if row[target] else np.nan)
        rows.append(rec)
    fig.tight_layout()
    return VizResult(fig, axes[:, 0], pd.DataFrame(rows))


def upset(
    sets: Mapping[str, Iterable[Any]] | pd.DataFrame,
    min_size: int = 1,
    max_intersections: int | None = 20,
    sort_by: Literal["size", "degree"] = "size",
    color: str = "#333333",
) -> VizResult:
    """UpSet plot of set intersections — a scalable replacement for Venn diagrams.

    Venn diagrams become unreadable beyond three sets. UpSet shows each
    *exclusive* intersection as a column: a bar for its size and a dot
    matrix for which sets take part (Lex et al., 2014).

    Parameters
    ----------
    sets
        A mapping ``{name: iterable of members}`` or a DataFrame of boolean
        membership columns (one row per element).
    min_size
        Hide intersections smaller than this.
    max_intersections
        Show at most this many, keeping the largest.
    sort_by
        Display order: ``"size"`` (largest first) or ``"degree"`` (number of
        sets involved, then size).

    References
    ----------
    Lex, A., Gehlenborg, N., Strobelt, H., Vuillemot, R., & Pfister, H. (2014).
    UpSet: visualization of intersecting sets. *IEEE TVCG*, 20(12), 1983–1992.
    """
    check_choice("sort_by", sort_by, ["size", "degree"])
    if isinstance(sets, pd.DataFrame):
        for c in sets.columns:
            bad = [v for v in pd.unique(sets[c].dropna()) if v not in (True, False)]  # 0/1 compare equal to bools
            if bad:
                raise ValueError(f"membership column {c!r} must hold True/False or 1/0 (missing = not a member); "
                                 f"found {bad[:5]}")
        membership = pd.DataFrame({c: [bool(v) if pd.notna(v) else False for v in sets[c]] for c in sets.columns},
                                  index=sets.index)
    else:
        sets = {k: set(v) for k, v in sets.items()}
        universe = sorted(set().union(*sets.values()), key=str)
        membership = pd.DataFrame({k: [e in s for e in universe] for k, s in sets.items()}, index=universe)
    names = list(membership.columns)
    if len(names) < 2:
        raise ValueError("need at least two sets")
    membership = membership[membership.any(axis=1)]
    combo = membership.apply(lambda r: tuple(r.to_numpy()), axis=1)
    counts = combo.value_counts()
    table = pd.DataFrame(list(counts.index), columns=names)
    table["size"] = counts.to_numpy()
    table["degree"] = table[names].sum(axis=1)
    table = table[table["size"] >= min_size]
    table = table.sort_values(["size", "degree"], ascending=[False, False], ignore_index=True)
    if max_intersections:  # keep the largest intersections, then apply the requested display order
        table = table.head(max_intersections)
    if sort_by == "degree":
        table = table.sort_values(["degree", "size"], ascending=[True, False], ignore_index=True)
    set_sizes = membership.sum().reindex(names)

    k, m = len(names), len(table)
    fig = plt.figure(figsize=(max(6, 0.45 * m + 3), 2.5 + 0.4 * k + 2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, max(m, 1) / 4], height_ratios=[3, 0.45 * k], wspace=0.05, hspace=0.05)
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_mat = fig.add_subplot(gs[1, 1], sharex=ax_bar)
    ax_set = fig.add_subplot(gs[1, 0], sharey=ax_mat)
    xs = np.arange(m)
    bars = ax_bar.bar(xs, table["size"], color=color, width=0.6)
    ax_bar.bar_label(bars, fontsize=8, padding=1)
    ax_bar.set_ylabel("Intersection size")
    ax_bar.spines[["top", "right", "bottom"]].set_visible(False)
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    for j, row in table.iterrows():
        on = [i for i, nme in enumerate(names) if row[nme]]
        ax_mat.scatter([j] * k, range(k), s=60, color="#dddddd", zorder=1)
        ax_mat.scatter([j] * len(on), on, s=60, color=color, zorder=2)
        if len(on) > 1:
            ax_mat.plot([j, j], [min(on), max(on)], color=color, lw=2, zorder=1)
    ax_mat.set_yticks(range(k), names)
    ax_mat.tick_params(axis="y", left=False, labelleft=False)
    ax_mat.set_xticks([])
    ax_mat.set_ylim(-0.6, k - 0.4)
    ax_mat.invert_yaxis()
    for s in ax_mat.spines.values():
        s.set_visible(False)
    ax_set.barh(range(k), set_sizes, color=NEUTRAL, height=0.5)
    ax_set.invert_xaxis()
    ax_set.yaxis.tick_right()
    ax_set.set_yticks(range(k), names)
    ax_set.tick_params(axis="y", length=0, pad=6)
    ax_set.set_xlabel("Set size")
    ax_set.spines[["top", "left", "right"]].set_visible(False)
    fig.add_subplot(gs[0, 0]).axis("off")
    return VizResult(fig, np.array([ax_bar, ax_mat, ax_set]), table, {"set_sizes": set_sizes})
