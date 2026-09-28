"""Part-to-whole charts: shares, flows between stages, and set overlaps."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from . import stats as st
from ._core import (
    NEUTRAL,
    OKABE_ITO,
    AbbrevFormatter,
    VizResult,
    _level_key,
    abbreviate,
    check_choice,
    check_dataframe,
    check_has_values,
    check_numeric,
    cleanup_on_error,
    column_list,
    get_ax,
    level_label,
    palette,
    select_columns,
    slot_colors,
    text_color,
)
from ._core import (
    category_order as _levels,
)

__all__ = ["waffle", "percent_grid", "stacked_percentages", "donut_grid", "nested_pie", "circular_bar",
           "waterfall", "funnel", "bullet", "upset"]


def _check_count(name: str, value: Any, minimum: int = 1) -> None:
    """Raise unless *value* is an integer of at least *minimum* (a grid size, a gap)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _missing_rows(data: pd.DataFrame, columns: Sequence[str], label: str | None = None) -> list[Any]:
    """Labels (index labels, or the *label* column) of the rows with a missing value in *columns*."""
    rows = select_columns(data, columns).isna().any(axis=1).to_numpy()
    return (data.index if label is None else data[label])[rows].tolist()


def _sum(values: pd.Series, by: pd.Series | None = None) -> Any:
    """Sum of *values* (per group of *by*), in float where an integer sum could wrap around."""
    total = values.sum() if by is None else values.groupby(by, sort=False, observed=True).sum()
    if pd.api.types.is_integer_dtype(values):
        # int64 sums wrap around silently; past 2**62 (a margin for float rounding) sum in float instead
        floats = pd.Series(values.to_numpy(dtype=float, na_value=np.nan), index=values.index)
        floats = floats.sum() if by is None else floats.groupby(by, sort=False, observed=True).sum()
        if (np.abs(floats) >= 2.0**62).any():
            return floats
    return total


def _aggregate(data: pd.DataFrame, category: str, value: str | None) -> pd.Series:
    if value is None:
        counts = data[category].value_counts(sort=False)
        return counts[counts > 0]  # a Categorical also counts its unused categories; drop them, as the sums do
    check_numeric(data, value)
    if (data[value] < 0).any():  # per row: summing first would net a negative row against the others
        raise ValueError("waffle values must be non-negative")
    return _sum(data[value], data[category])


@cleanup_on_error
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
    or under-fill it. With *value* omitted, rows are counted. Values must be
    non-negative; missing values count as zero, but *category* and *value*
    each need at least one non-missing value. Categories are drawn largest
    first unless *order* is given. *order* must list every category that
    occurs and may add unused ones (drawn with zero tiles); unused categories
    of a ``Categorical`` are left out otherwise. Rows with a missing
    *category* are not dropped: they are drawn last, in grey, as
    "(missing)", and count towards the total. *rows* and *columns* (the grid
    size) must be positive integers.
    """
    check_dataframe(data, [category, value])
    _check_count("rows", rows)
    _check_count("columns", columns)
    check_has_values(data, category, value)
    totals = _aggregate(data, category, value)
    unknown = data[category].isna()  # rows without a category still count towards the total
    unknown_total = int(unknown.sum()) if value is None else _sum(data.loc[unknown, value])
    if not ((totals > 0).any() or unknown_total > 0):  # not totals.sum(), which can overflow
        raise ValueError("waffle needs a positive total; every value is zero")
    if order is not None:  # must list every category, or the shares would be renormalized over a subset
        levels = _levels(data[category], order, complete=True, allow_absent=True)
    else:
        levels = list(totals.sort_values(ascending=False, kind="stable").index)
    values = totals.reindex(levels).fillna(0).to_numpy()
    cols = palette(len(levels), colors)
    names = [level_label(lvl) for lvl in levels]
    if unknown.any():
        levels, names, cols = levels + [np.nan], names + ["(missing)"], cols + [NEUTRAL]
        values = np.append(values, unknown_total)
    n_tiles = rows * columns
    tiles = st.largest_remainder(values.astype(float), n_tiles)

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
    # Rescale by a power of two (exact, as largest_remainder does) so the sum cannot overflow, e.g. near 1e308.
    scaled = np.ldexp(values.astype(float), -int(np.frexp(float(values.max()))[1]))
    share = scaled / scaled.sum() * 100
    ax.legend(handles=[Patch(color=c, label=f"{name}: {abbreviate(v)} ({p:.1f}%)")
                       for name, c, v, p in zip(names, cols, values, share)],
              frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title(f"1 tile ≈ {100 / n_tiles:.2g}% of total", loc="left", fontsize="small", color=NEUTRAL)
    table = pd.DataFrame({category: levels, "value": values, "percent": share, "tiles": tiles})
    return VizResult(fig, ax, table)


@cleanup_on_error
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
    groups are not over-read. A group with no non-missing values is titled
    "no data (n=0)". Each dot is 1 %; the percentage is rounded half up to
    a whole number of dots.

    Parameters
    ----------
    column
        A binary column (two distinct values, e.g. 0/1, True/False, "yes"/"no")
        with at least one non-missing value.
    success
        The value counted as success. Defaults to ``True``/``1`` for boolean or
        0/1 data; required otherwise. It must be one of the column's two
        values (a column with a single value cannot be checked, so a
        misspelled *success* there reads as 0 %).
    level
        Confidence level of the interval, strictly between 0 and 1.
    colors
        A tuple of two colours: ``(success, other)``.
    col_wrap
        Panels per row (a positive integer).

    Returns
    -------
    VizResult
        ``table`` has one row per panel: ``facet`` (the *facet* level; the
        column is left out without *facet*), ``n``, ``successes``,
        ``percent`` and the Wilson interval ``ci_low``/``ci_high``.
        ``percent``, ``ci_low`` and ``ci_high`` are **percentages** (0–100),
        unlike ``stats.wilson_ci`` and ``centered_bar``, whose intervals are
        proportions (0–1). A group with ``n = 0`` has ``nan`` for all three.
    """
    check_dataframe(data, [column, facet])
    _check_count("col_wrap", col_wrap)
    st._check_level(level)
    colors = slot_colors(colors, ("success", "other"))
    check_has_values(data, column)
    values = data[column].dropna()
    uniq = {_level_key(v) for v in pd.unique(values)}  # a datetime64 misses an equal Timestamp in a set on NumPy 1.x
    if len(uniq) > 2:
        raise ValueError(f"{column!r} must have at most two distinct values, found {len(uniq)}")
    if success is None:
        if uniq <= {0, 1}:
            success = 1
        else:
            raise ValueError(f"set success= to the value of {column!r} to count (found {sorted(map(str, uniq))})")
    elif len(uniq) == 2 and _level_key(success) not in uniq:
        raise ValueError(f"success={success!r} does not occur in {column!r}; its values are {sorted(map(str, uniq))}")
    facets = _levels(data[facet], facet_order) if facet is not None else [None]
    ncol = min(col_wrap, len(facets))
    nrow = int(np.ceil(len(facets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.6 * nrow), squeeze=False)
    rows = []
    for ax, f in zip(axes.flat, facets):
        s = (data if f is None else data[data[facet] == f])[column].dropna()
        n, k = int(s.size), int((s == success).sum())
        p = k / n if n else np.nan
        lo, hi = st.wilson_ci(k, n, level)
        filled = (200 * k + n) // (2 * n) if n else 0  # exact, halves round up (round() would go to even)
        xs, ys = np.meshgrid(np.arange(10), np.arange(10)[::-1])
        fill = np.arange(100) < filled
        ax.scatter(xs.ravel(), ys.ravel(), s=70, c=[colors[0] if on else colors[1] for on in fill])  # any colour form
        ax.set_xlim(-0.7, 9.7)
        ax.set_ylim(-0.7, 9.7)
        ax.set_aspect("equal")
        ax.axis("off")
        head = f"{level_label(f)}: " if f is not None else ""
        stat = f"{p:.1%}\n{level * 100:.10g}% CI {float(lo):.1%}–{float(hi):.1%}, n={n}" if n else "no data (n=0)"
        ax.set_title(f"{head}{stat}", fontsize="medium")
        rows.append({"facet": f, "n": n, "successes": k, "percent": 100 * p,
                     "ci_low": 100 * float(lo), "ci_high": 100 * float(hi)})
    for ax in list(axes.flat)[len(facets):]:
        ax.set_visible(False)
    fig.tight_layout()
    table = pd.DataFrame(rows)
    if facet is None:
        table = table.drop(columns="facet")
    return VizResult(fig, axes, table, {"success": success, "level": level})


@cleanup_on_error
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
    category_order_: Sequence[Any] | None = None,
) -> VizResult:
    """100 % stacked bars: the share of each *category* within each *group*.

    Group sizes (*n*) are shown in the tick labels because percentages hide
    them. Segments smaller than *min_label* percent are not labelled. Rows
    with a missing *category* are left out of *n* and the percentages.
    *group* and *category* must be different columns.

    *group_order* may list a subset of groups. *category_order* sets the
    stacking order; it must list every category that occurs (so shares add to
    100 %) and may add unused ones, such as the empty points of a response
    scale. A group with no non-missing answers is drawn empty, with ``n=0``
    and ``nan`` percentages in the table.

    ``category_order_`` is a deprecated alias of *category_order*.
    """
    if category_order_ is not None:
        import warnings

        # stacklevel 3: past cleanup_on_error's wrapper to the caller's line
        warnings.warn("category_order_ is deprecated; use category_order", FutureWarning, stacklevel=3)
        category_order = category_order if category_order is not None else category_order_
    check_dataframe(data, [group, category])
    if group == category:
        raise ValueError(f"group and category must be different columns; both are {group!r}")
    reserved = [c for c in (group, category) if c in ("n", "percent")]
    if reserved:
        raise ValueError(f"rename column(s) {reserved}: 'n' and 'percent' are the names of the output columns")
    groups = _levels(data[group], group_order)
    cats = _levels(data[category], category_order, complete=True, allow_absent=True)
    # crosstab aligns its inputs on the index, which fails on repeated labels (pd.concat) in pandas 3
    counts = pd.crosstab(data[group].reset_index(drop=True), data[category].reset_index(drop=True))
    counts = counts.reindex(index=groups, columns=cats, fill_value=0)
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    cols = palette(len(cats), colors)
    fig, ax = get_ax(ax, figsize=(9, max(3, 0.5 * len(groups) + 1.5)) if horizontal else (max(6, 0.9 * len(groups) + 3), 5))
    pos = np.arange(len(groups))
    base = np.zeros(len(groups))
    for c, col in zip(cats, cols):
        vals = pct[c].to_numpy()
        bars = (ax.barh(pos, vals, left=base, color=col, label=level_label(c), edgecolor="white")
                if horizontal else ax.bar(pos, vals, bottom=base, color=col, label=level_label(c), edgecolor="white"))
        ax.bar_label(bars, labels=[f"{v:.0f}%" if v >= min_label else "" for v in vals], label_type="center",
                     fontsize=8, color=text_color(col))
        base += vals
    ticks = [f"{level_label(g)}\n(n={n})" for g, n in zip(groups, counts.sum(axis=1))]
    if horizontal:
        ax.set_yticks(pos, ticks)
        ax.xaxis.set_major_formatter(PercentFormatter())
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
    else:
        ax.set_xticks(pos, ticks)
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_ylim(0, 100)
    ax.legend(title=str(category), frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.spines[["top", "right"]].set_visible(False)
    table = counts.stack().rename("n").reset_index()
    group_n = table.groupby(group)["n"].transform("sum")
    table["percent"] = table["n"] / group_n.replace(0, np.nan) * 100  # nan for a group with n = 0
    return VizResult(fig, ax, table)


@cleanup_on_error
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
    *columns* (a list of labels or any other iterable of them, e.g.
    ``df.columns[1:]``, but not a lone string) defaults to the numeric,
    non-boolean columns. Values must be non-negative and not missing (fill or
    drop missing cells first); an all-zero row is drawn as an empty grey
    ring. *col_wrap* (donuts per row) must be a positive integer.

    Note: people compare angles less accurately than lengths (Cleveland &
    McGill, 1984); for precise comparisons use :func:`stacked_percentages`.
    """
    columns = None if columns is None else column_list("columns", columns)
    check_dataframe(data, columns or [])
    _check_count("col_wrap", col_wrap)
    columns = columns if columns is not None else [
        c for c in data.columns if pd.api.types.is_numeric_dtype(data[c]) and not pd.api.types.is_bool_dtype(data[c])]
    if not columns:
        raise ValueError("donut_grid needs at least one category column: columns= is empty or data has no "
                         "numeric (non-boolean) columns")
    check_numeric(data, *columns)
    missing = _missing_rows(data, columns)
    if missing:
        raise ValueError(f"donut_grid needs a value in every cell of {columns}; row(s) {missing} have missing "
                         "values (fill them, e.g. with data.fillna(0), or drop those rows)")
    values = select_columns(data, columns).to_numpy(dtype=float)  # nullable Int64/Float64 too
    if (values < 0).any():
        raise ValueError("donut values must be non-negative")
    cols = palette(len(columns), colors)
    n = len(data)
    ncol = min(col_wrap, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol + 1.5, 3.3 * nrow), squeeze=False)
    for ax, idx, vals in zip(axes.flat, data.index, values):
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
        ax.set_title(level_label(idx), fontsize="medium")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.legend(handles=[Patch(color=c, label=level_label(k)) for k, c in zip(columns, cols)], frameon=False,
               loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    totals = values.sum(axis=1)
    share = values / np.where(totals == 0, np.nan, totals)[:, None] * 100  # nan for an all-zero row
    # Built column by column (not with melt), so data columns named "row" or "percent" cannot collide.
    row_labels = data.index.to_flat_index().take(np.tile(np.arange(n), len(columns)))  # keeps the index dtype
    table = pd.DataFrame({"row": row_labels, "category": pd.Index(columns).repeat(n),
                          "percent": share.ravel(order="F")})
    return VizResult(fig, axes, table)


@cleanup_on_error
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
    rows are counted. Values must be non-negative, with a positive total;
    missing values count as zero. *outer* and *inner* must be different
    columns, each with at least one non-missing value. Rows with a missing
    *outer* or *inner* category are not dropped: they are drawn in grey as
    "(missing)" (a missing *outer* last) and count towards the total.
    Neither category column may be named ``"value"``, ``"percent_of_total"``
    or ``"percent_of_parent"``, as those name the table's output columns.
    """
    check_dataframe(data, [outer, inner, value])
    if outer == inner:
        raise ValueError(f"outer and inner must be different columns; both are {outer!r}")
    reserved = [c for c in (outer, inner) if c in ("value", "percent_of_total", "percent_of_parent")]
    if reserved:
        raise ValueError(f"rename column(s) {reserved}: 'value', 'percent_of_total' and 'percent_of_parent' are the "
                         "names of the output columns")
    check_has_values(data, outer, inner)  # an all-missing value column is a zero total, reported below
    # dropna=False: rows with a missing category keep their share of the total
    grouped = data.groupby([outer, inner], sort=False, observed=True, dropna=False)
    if value is None:
        agg = grouped.size().rename("value")
    else:
        check_numeric(data, value)
        if (data[value] < 0).any():
            raise ValueError("nested_pie values must be non-negative")
        agg = grouped[value].sum().rename("value")
    agg = agg.reset_index()
    if not agg["value"].sum() > 0:
        raise ValueError("nested_pie needs a positive total; every value is zero or missing")
    parents = list(pd.unique(agg[outer].dropna()))
    base_cols = palette(len(parents), colors)
    # Match rows to parents by value (as codes), which also works for nullable dtypes; -1 marks a missing parent.
    codes = pd.Index(parents).get_indexer(agg[outer])
    if (codes < 0).any():  # drawn grey, after the others
        parents, base_cols = parents + [np.nan], base_cols + [NEUTRAL]
        codes = np.where(codes < 0, len(parents) - 1, codes)
    rank = np.argsort(codes, kind="stable")
    agg, codes = agg.iloc[rank].reset_index(drop=True), codes[rank]
    parent_tot = agg["value"].groupby(codes).sum()
    child_cols = []
    for i, c in enumerate(base_cols):
        rgb = np.array(to_rgb(c))
        child_cols += [tuple(rgb + (1 - rgb) * t) for t in np.linspace(0.25, 0.65, int((codes == i).sum()))]
    child_cols = [NEUTRAL if gap else c for c, gap in zip(child_cols, agg[inner].isna())]

    def name(v: Any) -> str:
        return "(missing)" if pd.isna(v) else level_label(v)

    fig, ax = get_ax(ax, figsize=(7, 7))
    wedges, texts = ax.pie(parent_tot, radius=0.7, colors=base_cols,
                           labels=[name(p) for p in parents] if labels else None,
                           labeldistance=0.75, wedgeprops={"width": 0.35, "edgecolor": "white"}, startangle=90,
                           counterclock=False, textprops={"weight": "bold", "ha": "center", "fontsize": 9})
    for t, c in zip(texts, base_cols):
        t.set_color(text_color(c))
    ax.pie(agg["value"], radius=1.0, colors=child_cols, labels=[name(v) for v in agg[inner]] if labels else None,
           labeldistance=1.06, wedgeprops={"width": 0.3, "edgecolor": "white"}, startangle=90, counterclock=False,
           textprops={"fontsize": 8})
    ax.set_aspect("equal")
    agg["percent_of_total"] = agg["value"] / agg["value"].sum() * 100
    agg["percent_of_parent"] = agg["value"] / agg["value"].groupby(codes).transform("sum") * 100
    return VizResult(fig, ax, agg)


@cleanup_on_error
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
    not precise comparison. Values must be non-negative; a missing value
    leaves an empty (labelled) slot, but *label* and *value* each need at
    least one non-missing value. Rows with a missing *group* are drawn in
    grey as a final "(missing)" cluster, not dropped (all of them, if the
    *group* column has no values).

    Parameters
    ----------
    gap
        Empty slots between groups (a non-negative integer).
    ax
        A **polar** Axes to draw into, e.g. ``plt.subplot(projection='polar')``
        or ``plt.subplots(subplot_kw={'projection': 'polar'})``; any other
        Axes raises ``ValueError`` before anything is drawn. By default a new
        figure is created.
    """
    if ax is not None and getattr(ax, "name", None) != "polar":
        raise ValueError("circular_bar needs a polar Axes, e.g. plt.subplot(projection='polar')")
    check_dataframe(data, [label, value, group])
    check_numeric(data, value)
    check_has_values(data, label, value)
    _check_count("gap", gap, minimum=0)
    if (data[value] < 0).any():
        raise ValueError("circular_bar values must be non-negative")
    # dict.fromkeys: one column may play two roles (label="city", group="city"), but must be selected once
    d = data[list(dict.fromkeys([label, value] + ([group] if group is not None else [])))].copy()
    # a group column with no values: every row goes to the "(missing)" cluster below
    groups = (_levels(d[group]) if d[group].notna().any() else []) if group is not None else [None]
    cols = palette(len(groups), colors)
    # Match rows to groups by value (as codes), which also works for nullable dtypes; -1 marks a missing group.
    codes = pd.Index(groups).get_indexer(d[group]) if group is not None else np.zeros(len(d), dtype=int)
    if (codes < 0).any():  # rows without a group are drawn grey, not dropped
        groups, cols = groups + [None], cols + [NEUTRAL]
        codes = np.where(codes < 0, len(groups) - 1, codes)
    parts, part_codes, slot_idx, slot = [], [], [], 0
    for i in range(len(groups)):
        sub = d[codes == i]
        if sort:
            sub = sub.sort_values(value, ascending=False)
        parts.append(sub)
        part_codes.append(np.full(len(sub), i))
        slot_idx += range(slot, slot + len(sub))
        slot += len(sub) + (gap if group is not None else 0)  # empty slots separate the groups
    ordered = pd.concat(parts, ignore_index=True)
    ordered_codes = np.concatenate(part_codes)
    n_slots = max(slot, 1)
    theta = np.linspace(0, 2 * np.pi, n_slots, endpoint=False)[slot_idx]
    vals = ordered[value].to_numpy(dtype=float, na_value=np.nan)  # nullable Int64/Float64 with pd.NA too
    drawn = np.isfinite(vals)  # a missing value leaves its slot empty: Matplotlib cannot render a NaN polar bar
    top = float(vals[drawn].max()) if drawn.any() else 0.0
    vmax = top or 1.0  # all-zero values: any divisor draws empty bars
    heights = np.where(drawn, vals, 0.0) / vmax
    col_for = [cols[c] for c in ordered_codes]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    else:
        fig = ax.figure
    width = 2 * np.pi / n_slots * 0.9
    ax.bar(theta[drawn], heights[drawn], width=width, bottom=inner_radius,
           color=[c for c, k in zip(col_for, drawn) if k], edgecolor="white", lw=0.5)
    for t, h, name in zip(theta, heights, ordered[label]):
        # The axis runs clockwise from north, so the on-screen angle is 90° − θ.
        deg = (90 - np.rad2deg(t)) % 360
        flip = 90 < deg < 270  # keep labels on the left half upright
        ax.text(t, inner_radius + h + 0.03, level_label(name), rotation=deg + 180 if flip else deg,
                rotation_mode="anchor", ha="right" if flip else "left", va="center", fontsize=7)
    if group is not None:
        for i, (g, c) in enumerate(zip(groups, cols)):
            th = theta[ordered_codes == i]
            ax.plot(np.linspace(th.min(), th.max(), 30), np.full(30, inner_radius - 0.05), color=c, lw=2)
            ax.text(th.mean(), inner_radius - 0.16, "(missing)" if g is None else level_label(g), ha="center",
                    va="center", fontsize=9, weight="bold")
    ax.set_ylim(0, inner_radius + 1.35)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.axis("off")
    ax.set_title(f"{value} (bar length relative to max = {abbreviate(top)})", fontsize="medium")
    return VizResult(fig, ax, ordered)


@cleanup_on_error
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
    label
        Column naming each step; it needs at least one non-missing value.
    value
        Numeric column; every row needs a value, since one gap would make
        every later running total unknown.
    values_are
        ``"changes"``: each row is an increment (the first row may be the
        starting value, see *start_label*). ``"levels"``: each row is a running
        level and increments are computed as differences.
    start_label
        Treat the first row as the starting total, drawn as a total bar with
        this label. With ``values_are="levels"`` it defaults to the first
        row's label.
    total_label
        Label for a final total bar; ``None`` omits it.
    colors
        A tuple of three colours: ``(increase, decrease, total)``.
    """
    check_dataframe(data, [label, value])
    check_choice("values_are", values_are, ["changes", "levels"])
    check_numeric(data, value)
    colors = slot_colors(colors, ("increase", "decrease", "total"))
    check_has_values(data, label)
    missing = _missing_rows(data, [value], label)
    if missing:
        raise ValueError(f"{value!r} is missing for {missing}; a waterfall needs every step "
                         "(fill or drop those rows first)")
    labels = data[label].astype(str).tolist()
    vals = data[value].to_numpy(dtype=float)
    if values_are == "levels":
        changes = np.r_[vals[0], np.diff(vals)]
        if start_label is None:
            start_label = labels[0]
    else:
        changes = vals.copy()
    if start_label is not None:
        labels[0] = str(start_label)
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
    ax.yaxis.set_major_formatter(AbbrevFormatter())
    ax.set_ylabel(value)
    ax.legend(handles=[Patch(color=colors[0], label="increase"), Patch(color=colors[1], label="decrease"),
                       Patch(color=colors[2], label="total")], frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax.spines[["top", "right"]].set_visible(False)
    return VizResult(fig, ax, table)


@cleanup_on_error
def funnel(
    data: pd.DataFrame,
    stage: str,
    value: str,
    color: str = OKABE_ITO[0],
    ax: Axes | None = None,
) -> VizResult:
    """Conversion funnel with step-to-step and overall conversion rates.

    Stages are drawn in row order. The table reports ``pct_of_first`` and
    ``pct_of_previous`` (step conversion) and the ``drop_off`` count, so
    *stage* and *value* may not use those names; they must also be different
    columns. Every stage needs a non-negative, non-missing value, and *stage*
    needs at least one non-missing name. A rate after a zero stage is
    undefined: it is ``NaN`` in the table and shown as "–" on the chart.
    The chart labels rates in whole percent, with more digits where that
    would show a non-zero rate as 0 % or a rate short of 100 % as 100 %.
    """
    check_dataframe(data, [stage, value])
    reserved = [c for c in (stage, value) if c in ("pct_of_first", "pct_of_previous", "drop_off")]
    if reserved:
        raise ValueError(f"rename column(s) {reserved}: 'pct_of_first', 'pct_of_previous' and 'drop_off' are the "
                         "names of the output columns")
    if stage == value:
        raise ValueError(f"stage and value must be different columns; both are {stage!r}")
    check_numeric(data, value)
    check_has_values(data, stage, value)
    missing = _missing_rows(data, [value], stage)
    if missing:
        raise ValueError(f"{value!r} is missing for stage(s) {missing}; a funnel needs a value for every stage")
    if (data[value] < 0).any():
        raise ValueError("funnel values must be non-negative")
    t = data[[stage, value]].reset_index(drop=True).copy()
    v = t[value].to_numpy(dtype=float)
    t["pct_of_first"] = v / v[0] * 100 if len(v) and v[0] else np.nan
    t["pct_of_previous"] = np.r_[100.0, v[1:] / np.where(v[:-1] == 0, np.nan, v[:-1]) * 100] if len(v) else []
    t["drop_off"] = np.r_[0.0, v[:-1] - v[1:]] if len(v) else []
    vmax = float(v.max()) or 1.0  # all-zero stages still get a usable axis
    fig, ax = get_ax(ax, figsize=(8, 0.7 * len(t) + 1.5))
    pos = np.arange(len(t))[::-1]
    ax.barh(pos, v, left=-v / 2, color=color, height=0.75)

    def pct(x: float) -> str:  # a rate after a zero stage is undefined
        if np.isnan(x):
            return "–"
        if 0 < x < 1:  # a non-zero rate never reads 0%
            return f"{abbreviate(x)}%"
        digits = 0  # whole percent, but a rate just off 100% never reads 100%
        while x != 100 and float(f"{x:.{digits}f}") == 100:
            digits += 1
        return f"{x:.{digits}f}%"

    for p, row in zip(pos, t.itertuples(index=False)):
        vv, pf, pp = row[1], row[2], row[3]
        text = f"{abbreviate(vv)}  ({pct(pf)})"
        if vv >= 0.3 * vmax:
            ax.text(0, p, text, ha="center", va="center", color=text_color(color), weight="bold", fontsize=9)
        else:  # too narrow to hold the label: put it beside the bar
            ax.text(vv / 2 + vmax * 0.02, p, text, ha="left", va="center", color="black", weight="bold", fontsize=9)
        if p != pos[0]:
            ax.text(vmax / 2 * 1.02, p + 0.5, f"↓ {pct(pp)} of previous", va="center", fontsize=8, color=NEUTRAL)
    ax.set_yticks(pos, t[stage].astype(str))
    ax.set_xlim(-vmax / 2 * 1.05, vmax / 2 * 1.6)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    return VizResult(fig, ax, t)


@cleanup_on_error
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

    Each row of *data* gets its own panel (and its own scale) in a new
    figure; a single row can instead be drawn into *ax*. A missing value,
    target or band limit is left out of its panel, but *label* and *value*
    each need at least one non-missing value.

    Parameters
    ----------
    bands
        Either a list of column names holding each row's band upper limits,
        or a list of finite numeric limits shared by every row; not a mix of
        the two, and not a lone string (use ``["good"]`` for one column).
        Ascending order. Entries that are all labels in ``data.columns``
        (e.g. ``[2, 3]`` for integer-labelled columns) are column names.
    ax
        An Axes to draw into. Only for single-row *data* (one bullet graph);
        with more rows it raises ``ValueError``, since each row needs its own
        panel. By default a new figure is created.

    Returns
    -------
    VizResult
        ``axes`` is an array with one Axes per row (the given *ax* for a
        single row); ``table`` has *label* and the ``value``, ``target`` and
        ``pct_of_target`` per row, so *label* may not use those names.

    References
    ----------
    Few, S. (2013). *Bullet Graph Design Specification*. Perceptual Edge.
    """
    bands = [] if bands is None else column_list("bands", bands)  # also accepts NumPy arrays and Series
    data = check_dataframe(data)
    names = [b for b in bands if isinstance(b, str) or b in data.columns]  # numbers may label columns too
    band_cols = names if len(names) == len(bands) else [b for b in bands if isinstance(b, str)]
    if band_cols and len(band_cols) < len(bands):
        raise ValueError("bands must be a list of column names or a list of numbers, not a mix of both; "
                         f"got {bands!r}")
    if not band_cols and not all(isinstance(b, numbers.Real) and math.isfinite(b) for b in bands):
        raise ValueError(f"bands must be a list of column names or of finite numbers, got {bands!r}")
    check_dataframe(data, [label, value, target, *band_cols])
    reserved = ["value"] + (["target", "pct_of_target"] if target is not None else [])
    if label in reserved:
        raise ValueError(f"rename the label column {label!r}: the table uses {reserved} for its output columns")
    check_numeric(data, value, *([target] if target is not None else []), *band_cols)
    check_has_values(data, label, value)
    n = len(data)
    if ax is not None and n != 1:
        raise ValueError(f"bullet draws one panel per row, so ax= needs data with a single row, got {n} rows; "
                         "select one row or leave out ax to get a figure with a panel per row")

    def floats(col: str) -> np.ndarray:  # nullable Int64/Float64 too; pd.NA becomes nan
        return data[col].to_numpy(dtype=float, na_value=np.nan)

    vals = floats(value)
    targets = floats(target) if target is not None else None
    limit_rows = (np.column_stack([floats(b) for b in band_cols]) if band_cols
                  else np.tile(np.asarray(bands, dtype=float), (n, 1)))
    own_figure = ax is None
    if own_figure:
        fig, axes = plt.subplots(n, 1, figsize=(8, 0.9 * n + 0.6), squeeze=False)
        axes = axes[:, 0]
    else:
        fig, axes = ax.figure, np.array([ax])
    rows = []
    for i, (ax, (_, row)) in enumerate(zip(axes, data.iterrows())):
        limits = [lim for lim in limit_rows[i].tolist() if not math.isnan(lim)]  # a missing limit: no band
        points = limits + [vals[i]] + ([targets[i]] if target is not None else [])
        top, bottom = max([0.0] + points), min([0.0] + points)  # the zero baseline is always in view
        if top == bottom:
            top = 1.0
        greys = plt.get_cmap("Greys")(np.linspace(0.45, 0.15, max(len(limits), 1)))
        prev = bottom  # the first band runs from the start of the axis up to its limit
        for lim, g in zip(sorted(limits), greys):
            ax.barh(0, lim - prev, left=prev, height=1, color=g)
            prev = lim
        ax.barh(0, vals[i], height=0.35, color=bar_color)
        if target is not None:
            ax.plot([targets[i]] * 2, [-0.35, 0.35], color="black", lw=2.5)
        pad = 0.02 * (top - bottom)
        ax.set_xlim(bottom - (pad if bottom < 0 else 0), top + pad)
        if bottom < 0:
            ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks([0], [level_label(row[label])])
        ax.set_ylim(-0.5, 0.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        rec = {label: row[label], "value": row[value]}
        if target is not None:
            rec.update(target=row[target], pct_of_target=vals[i] / targets[i] * 100 if targets[i] else np.nan)
        rows.append(rec)
    if own_figure:  # leave the layout of the caller's figure alone
        fig.tight_layout()
    return VizResult(fig, axes, pd.DataFrame(rows))


@cleanup_on_error
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
        membership columns (one row per element). Missing values (``None``,
        ``NaN``, ``NaT``, ``pd.NA``) are not members, in either form. One set
        or more, with at least one member between them; ``"size"`` and
        ``"degree"`` cannot be set names, as they name the table's count
        columns.
    min_size
        Hide intersections smaller than this (a number).
    max_intersections
        Show at most this many, keeping the largest (a positive integer;
        ``None`` shows all).
    sort_by
        Display order: ``"size"`` (largest first) or ``"degree"`` (number of
        sets involved, then size).

    References
    ----------
    Lex, A., Gehlenborg, N., Strobelt, H., Vuillemot, R., & Pfister, H. (2014).
    UpSet: visualization of intersecting sets. *IEEE TVCG*, 20(12), 1983–1992.
    """
    check_choice("sort_by", sort_by, ["size", "degree"])
    if not (isinstance(min_size, numbers.Real) and math.isfinite(min_size)):
        raise ValueError(f"min_size must be a finite number, got {min_size!r}")
    if max_intersections is not None:
        _check_count("max_intersections", max_intersections)
    if isinstance(sets, pd.DataFrame):
        for c in sets.columns:
            bad = [v for v in pd.unique(sets[c].dropna()) if v not in (True, False)]  # 0/1 compare equal to bools
            if bad:
                raise ValueError(f"membership column {c!r} must hold True/False or 1/0 (missing = not a member); "
                                 f"found {bad[:5]}")
        membership = pd.DataFrame({c: [bool(v) if pd.notna(v) else False for v in sets[c]] for c in sets.columns},
                                  index=sets.index)
    else:
        # A missing value is not a member, as in a DataFrame (one shared NaN would be a fake common member).
        sets = {k: {e for e in v if not (pd.api.types.is_scalar(e) and pd.isna(e))} for k, v in sets.items()}
        universe = sorted(set().union(*sets.values()), key=str)
        membership = pd.DataFrame({k: [e in s for e in universe] for k, s in sets.items()}, index=universe)
    names = list(membership.columns)
    if not names:
        raise ValueError("need at least one set")
    reserved = [nme for nme in names if nme in ("size", "degree")]
    if reserved:  # the table's count columns would overwrite these membership columns
        raise ValueError(f"rename set(s) {reserved}: 'size' and 'degree' are the names of the output columns")
    membership = membership[membership.any(axis=1)]
    if membership.empty:
        raise ValueError("upset needs at least one member; every set is empty")
    combo = membership.apply(lambda r: tuple(r.to_numpy()), axis=1)
    counts = combo.value_counts()
    table = pd.DataFrame(list(counts.index), columns=names)
    table["size"] = counts.to_numpy()
    table["degree"] = select_columns(table, names).sum(axis=1)  # by label: False/True set names are not a mask
    table = table[table["size"] >= min_size]
    table = table.sort_values(["size", "degree"], ascending=[False, False], ignore_index=True)
    if max_intersections is not None:  # keep the largest intersections, then apply the requested display order
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
    for j, row in enumerate(select_columns(table, names).to_numpy()):
        on = [i for i, member in enumerate(row) if member]
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
