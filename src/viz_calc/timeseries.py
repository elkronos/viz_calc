"""Time: period summaries, crossovers, calendars, timelines and animation."""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from ._core import (
    NEUTRAL,
    OKABE_ITO,
    AbbrevFormatter,
    VizResult,
    category_order,
    check_choice,
    check_dataframe,
    check_numeric,
    get_ax,
    palette,
)

__all__ = ["period_bars", "timeseries_fill", "calendar_heatmap", "gantt", "duration_plot", "animated_bubble"]

_PERIOD = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}


def _dates(s: pd.Series, name: str, keep_tz: bool = False) -> pd.Series:
    """Parse *s* to datetimes, strictly.

    Accepts datetime columns, strings pandas can parse with one consistent
    format, and ISO-8601 strings that mix date-only and date-time values or
    carry different UTC offsets (e.g. across a daylight-saving change).
    Ambiguous mixtures (such as day-first and month-first strings) raise
    instead of being guessed element by element.

    With ``keep_tz=False`` timezone-aware values become naive local wall-clock
    times, so each value lands on the calendar day it was recorded in. With
    ``keep_tz=True`` time zones are kept (mixed offsets become UTC), so
    elapsed times and instants stay exact.
    """
    import warnings

    def attempt(**kwargs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)  # pandas 2.x warns about mixed offsets
                out = pd.to_datetime(s, **kwargs)
        except (ValueError, TypeError, OverflowError):
            return None
        return out if pd.api.types.is_datetime64_any_dtype(out) else None

    if s.notna().sum() == 0:  # e.g. a blank CSV column read as float NaN: nothing to parse
        return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    values = s.cat.categories if isinstance(s.dtype, pd.CategoricalDtype) else s
    numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
    if not numeric and values.dtype == object:  # any number among dates, e.g. an Excel serial in one cell
        numeric = pd.Series(values).dropna().map(
            lambda v: isinstance(v, numbers.Number) and not isinstance(v, (bool, np.bool_))).any()
    if numeric:
        raise ValueError(f"column {name!r} holds numbers, which are ambiguous as dates (years? epoch seconds?); "
                         "convert it first, e.g. pd.to_datetime(values, unit='s') or format='%Y'")
    out = s if pd.api.types.is_datetime64_any_dtype(s) else attempt()
    if out is None:
        out = attempt(format="ISO8601")  # date-only and date-time ISO strings together
    if out is None:
        utc = attempt(format="ISO8601", utc=True)  # ISO strings with different UTC offsets
        if utc is None:
            raise ValueError(f"column {name!r} could not be parsed as dates; use one consistent format, "
                             "ideally ISO 8601 (YYYY-MM-DD or YYYY-MM-DD HH:MM[+HH:MM])")
        if keep_tz:
            has_zone = s[utc.notna()].map(lambda v: pd.Timestamp(v).tzinfo is not None)  # blanks don't count
            if not has_zone.all():  # an offset-less value would silently be read as UTC
                raise ValueError(f"column {name!r} mixes values with and without a UTC offset; make them consistent")
            out = utc
        else:  # the strings are valid ISO 8601, so element-wise parsing is unambiguous
            out = pd.to_datetime(s.map(lambda v: pd.NaT if pd.isna(v) else pd.Timestamp(v).tz_localize(None)))
    if not keep_tz and getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out


def _date_axis(ax: Axes, axis: str = "x", tz: Any = None) -> None:
    locator = mdates.AutoDateLocator(tz=tz)  # label ticks in the data's own time zone
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(locator)
    target.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=tz))


def period_bars(
    data: pd.DataFrame,
    date: str,
    value: str,
    freq: Literal["day", "week", "month", "quarter", "year"] = "month",
    stat: Literal["sum", "mean", "median", "count", "max", "min"] = "sum",
    trend_window: int | None = 3,
    color: str = OKABE_ITO[0],
    ax: Axes | None = None,
) -> VizResult:
    """Aggregate a series into calendar periods, with a moving-average trend.

    Bars show *stat* of *value* per period; the line is the trailing moving
    average of the bars over *trend_window* periods. Periods with no data are
    kept (as zero for sums/counts, a gap otherwise), so the time axis is
    honest. Calendar periods are used (a month is a month, not 30 days).
    """
    check_dataframe(data, [date, value])
    check_numeric(data, value)
    check_choice("freq", freq, list(_PERIOD))
    check_choice("stat", stat, ["sum", "mean", "median", "count", "max", "min"])
    d = pd.DataFrame({"period": _dates(data[date], date).dt.to_period(_PERIOD[freq]), "v": data[value]}).dropna(subset=["period"])
    if d.empty:
        raise ValueError("no dated rows")
    grouped = d.groupby("period")["v"]
    agg = pd.DataFrame({"value": grouped.agg(stat), "n": grouped.count()})
    full = pd.period_range(agg.index.min(), agg.index.max(), freq=agg.index.freq)
    agg = agg.reindex(full)
    agg["n"] = agg["n"].fillna(0).astype(int)
    if stat in ("sum", "count"):
        agg["value"] = agg["value"].fillna(0)
    if trend_window:
        agg["trend"] = agg["value"].rolling(trend_window, min_periods=1).mean()
    starts = agg.index.start_time
    widths = (agg.index.end_time - starts).total_seconds() / 86400 * 0.8

    fig, ax = get_ax(ax, figsize=(10, 4.5))
    ax.bar(starts, agg["value"], width=widths, align="edge", color=color, alpha=0.85, label=f"{stat} per {freq}")
    if trend_window:
        mids = starts + pd.to_timedelta(widths / 2, unit="D")
        ax.plot(mids, agg["trend"], color=OKABE_ITO[5], lw=2, label=f"{trend_window}-{freq} moving average")
    _date_axis(ax)
    ax.yaxis.set_major_formatter(AbbrevFormatter())
    ax.set_ylabel(f"{stat} of {value}")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    table = agg.rename_axis("period").reset_index()
    table["period_start"] = starts
    return VizResult(fig, ax, table, {"freq": freq, "stat": stat})


def timeseries_fill(
    data: pd.DataFrame,
    time: str,
    series: Sequence[str],
    labels: Sequence[str] | None = None,
    colors: tuple[str, str] = (OKABE_ITO[0], OKABE_ITO[5]),
    alpha: float = 0.25,
    ax: Axes | None = None,
) -> VizResult:
    """Two series with the gap between them shaded by which one leads.

    Shading is interpolated at the crossing points, so the colour switches
    exactly where the lines cross. Duplicate time stamps are averaged. The
    ``info`` reports the number of crossovers.
    """
    if len(series) != 2:
        raise ValueError("timeseries_fill needs exactly two series")
    a, b = series
    check_dataframe(data, [time, a, b])
    check_numeric(data, a, b)
    d = data[[time, a, b]].copy()
    col = d[time]
    if isinstance(col.dtype, pd.CategoricalDtype):  # judge a Categorical by its values (NaN-safe)
        col = col.astype(object)
    if not pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_datetime64_any_dtype(col):
        kind = pd.api.types.infer_dtype(col, skipna=True)
        numeric = None
        if kind in ("integer", "floating", "mixed-integer-float", "decimal", "mixed-integer"):
            try:  # e.g. years stored as objects, possibly some as text: keep them numeric
                numeric = pd.to_numeric(col)
            except (ValueError, TypeError):
                numeric = None
        # Parse dates before grouping so they sort chronologically, and keep time zones so the two
        # readings in the repeated hour of a daylight-saving change are not merged.
        col = numeric if numeric is not None else _dates(col, time, keep_tz=True)
    if pd.api.types.is_extension_array_dtype(col) and pd.api.types.is_numeric_dtype(col):
        # nullable Int64/Float64 -> numpy, which Matplotlib 3.7 needs; keep integers exact when nothing is missing
        col = col.astype("int64") if pd.api.types.is_integer_dtype(col) and col.notna().all() \
            else pd.Series(col.to_numpy(dtype=float, na_value=np.nan), index=col.index)
    d[time] = col
    d = d.dropna(subset=[time]).groupby(time, as_index=False).mean().sort_values(time)
    t = d[time]
    ya = d[a].to_numpy(dtype=float, na_value=np.nan)  # nullable dtypes with NA on pandas 2.0
    yb = d[b].to_numpy(dtype=float, na_value=np.nan)
    labels = list(labels) if labels else [a, b]
    fig, ax = get_ax(ax, figsize=(10, 4.5))
    ax.plot(t, ya, color=colors[0], lw=1.8, label=labels[0])
    ax.plot(t, yb, color=colors[1], lw=1.8, label=labels[1])
    ax.fill_between(t, ya, yb, where=ya >= yb, interpolate=True, color=colors[0], alpha=alpha, lw=0)
    ax.fill_between(t, ya, yb, where=ya < yb, interpolate=True, color=colors[1], alpha=alpha, lw=0)
    if pd.api.types.is_datetime64_any_dtype(t):
        _date_axis(ax, tz=getattr(t.dt, "tz", None))
    ax.set_xlabel(time)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    diff = ya - yb
    sign = np.sign(diff[~np.isnan(diff)])
    sign = sign[sign != 0]
    crossings = int(np.sum(sign[1:] != sign[:-1]))
    table = pd.DataFrame({time: d[time].to_numpy(), a: ya, b: yb, "difference": diff,
                          "leader": np.where(np.isnan(diff), None, np.where(diff >= 0, labels[0], labels[1]))})
    return VizResult(fig, ax, table, {"crossovers": crossings})


def calendar_heatmap(
    data: pd.DataFrame,
    date: str,
    value: str | None = None,
    stat: Literal["sum", "mean", "count", "max", "min"] = "sum",
    cmap: str = "viridis",
    missing_color: str = "#eeeeee",
) -> VizResult:
    """Calendar heatmap (one row of weeks per year), like a contribution graph.

    Days are aggregated with *stat* (count rows when *value* is omitted).
    Days without data are drawn in *missing_color* so they are not confused
    with zero. One colour scale is shared across years.
    """
    check_dataframe(data, [date, value])
    d = pd.DataFrame({"day": _dates(data[date], date).dt.normalize()})  # tz-aware dates: local calendar day
    if value is None:
        daily = d.groupby("day").size().astype(float)
        stat = "count"
    else:
        check_numeric(data, value)
        d["v"] = data[value].to_numpy()
        grouped = d.groupby("day")["v"]
        daily = grouped.agg(stat).astype(float)
        if stat != "count":
            daily[grouped.count() == 0] = np.nan  # a day whose values are all missing is "no data", not 0
    if daily.empty or daily.isna().all():
        raise ValueError("no dated rows with data")
    years = list(range(daily.index.min().year, daily.index.max().year + 1))
    vmin, vmax = float(np.nanmin(daily)), float(np.nanmax(daily))
    fig, axes = plt.subplots(len(years), 1, figsize=(12, 2.1 * len(years) + 0.4), squeeze=False)
    cm = plt.get_cmap(cmap).with_extremes(bad=missing_color)
    for ax, year in zip(axes[:, 0], years):
        days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        vals = daily.reindex(days)
        first_monday = days[0] - pd.Timedelta(days=days[0].weekday())
        week = ((days - first_monday).days // 7).to_numpy()
        grid = np.full((7, week.max() + 1), np.nan)
        grid[days.weekday, week] = vals.to_numpy()
        mesh = ax.pcolormesh(np.ma.masked_invalid(grid), cmap=cm, vmin=vmin, vmax=vmax, edgecolors="white", linewidth=1)
        ax.set_facecolor(missing_color)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_yticks([0.5, 2.5, 4.5], ["Mon", "Wed", "Fri"], fontsize=8)
        month_starts = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
        ax.set_xticks(((month_starts - first_monday).days // 7) + 0.5, month_starts.strftime("%b"), fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_ylabel(str(year), rotation=0, ha="right", va="center", fontsize=11, weight="bold")
    fig.colorbar(mesh, ax=axes[:, 0].tolist(), orientation="horizontal", fraction=0.04, pad=0.08,
                 label=f"{stat}{f' of {value}' if value is not None else ''} per day")
    return VizResult(fig, axes[:, 0], daily.rename("value").rename_axis("date").reset_index())


def _instant(t: Any) -> Any:
    """Matplotlib date numbers are UTC: convert tz-aware values to naive UTC (naive values are left alone)."""
    if isinstance(t, pd.Series):
        return t.dt.tz_convert("UTC").dt.tz_localize(None) if getattr(t.dt, "tz", None) is not None else t
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def _timeline_columns(d: pd.DataFrame, columns: Sequence[str]) -> Any:
    """Parse timeline columns keeping time zones; return the zone used to label the axis."""
    for c in columns:
        d[c] = _dates(d[c], c, keep_tz=True)
    used = [c for c in columns if d[c].notna().any()]  # an all-empty column has no zone to compare
    zones = [getattr(d[c].dt, "tz", None) for c in used]
    if any(z is None for z in zones) and any(z is not None for z in zones):
        raise ValueError(f"columns {list(columns)} mix timezone-aware and naive dates; make them consistent")
    tz = next((z for z in zones if z is not None), None)
    for c in columns:  # give empty columns the common zone (or none) so the subtractions work
        if c not in used:
            own = getattr(d[c].dt, "tz", None)
            if tz is not None:
                d[c] = d[c].dt.tz_localize(tz) if own is None else d[c].dt.tz_convert(tz)
            elif own is not None:
                d[c] = d[c].dt.tz_localize(None)
    return tz


def _timeline(ax: Axes, labels: list[str], starts: pd.Series, ends: pd.Series, colors: list[str], height: float,
              alpha: float = 1.0, zorder: int = 2) -> None:
    pos = np.arange(len(labels))[::-1]
    widths = (ends - starts).dt.total_seconds() / 86400  # true elapsed time, across time zones and DST
    ax.barh(pos, widths, left=mdates.date2num(_instant(starts)), height=height, color=colors, alpha=alpha,
            zorder=zorder)


def gantt(
    data: pd.DataFrame,
    task: str,
    start: str,
    end: str,
    group: str | None = None,
    sort: bool = True,
    today: Any = None,
    colors: Sequence[str] | str | None = None,
    ax: Axes | None = None,
) -> VizResult:
    """Gantt chart of tasks between start and end dates, coloured by *group*.

    Tasks are sorted by start date (set ``sort=False`` to keep row order).
    The date axis adapts its tick spacing to the range. Pass ``today`` (a date
    or ``"now"``) to draw a reference line.
    """
    check_dataframe(data, [task, start, end, group])
    d = data.copy()
    tz = _timeline_columns(d, [start, end])
    bad = d[d[end] < d[start]]
    if len(bad):
        raise ValueError(f"end is before start for task(s): {bad[task].tolist()}")
    if sort:
        d = d.sort_values(start, kind="stable")
    groups = category_order(d[group]) if group is not None else [None]
    cols = palette(len(groups), colors)
    # Match by value (not hashing), so datetime-like groups work; a missing group is drawn grey.
    codes = pd.Categorical(d[group], categories=groups).codes if group is not None else np.zeros(len(d), dtype=int)
    col = [cols[c] if c >= 0 else NEUTRAL for c in codes]
    fig, ax = get_ax(ax, figsize=(10, 0.45 * len(d) + 1.5))
    _timeline(ax, d[task].astype(str).tolist(), d[start], d[end], col, 0.6)
    ax.set_yticks(np.arange(len(d))[::-1], d[task].astype(str))
    ax.xaxis_date()
    _date_axis(ax, tz=tz)
    if today is not None:
        t = pd.Timestamp.now(tz=tz) if today == "now" else pd.Timestamp(today)
        if tz is not None and t.tzinfo is None:
            # a plain date/time means local time in the data's zone; on DST-change days pick a valid instant
            t = t.tz_localize(tz, nonexistent="shift_forward", ambiguous=False)
        elif tz is None and t.tzinfo is not None:
            t = t.tz_localize(None)
        ax.axvline(mdates.date2num(_instant(t)), color=OKABE_ITO[5], lw=1.5, ls="--")
    if group is not None:
        handles = [Patch(color=c, label=str(g)) for g, c in zip(groups, cols)]
        if (codes < 0).any():
            handles.append(Patch(color=NEUTRAL, label="(missing)"))
        ax.legend(handles=handles, title=str(group), frameon=False,
                  loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    table = d[[task, start, end] + ([group] if group is not None else [])].reset_index(drop=True)
    table["duration_days"] = (table[end] - table[start]).dt.total_seconds() / 86400
    return VizResult(fig, ax, table)


def duration_plot(
    data: pd.DataFrame,
    label: str,
    start: str,
    end: str,
    inner_start: str,
    inner_end: str,
    outer_name: str = "Overall period",
    inner_name: str = "Active duration",
    sort: bool = True,
    colors: tuple[str, str] = ("#a6bddb", OKABE_ITO[1]),
    ax: Axes | None = None,
) -> VizResult:
    """Each row's overall window with an inner duration drawn inside it.

    Example: a contract period with the time actually worked. The table
    reports both lengths in days and the inner share of the window.
    """
    check_dataframe(data, [label, start, end, inner_start, inner_end])
    d = data.copy()
    tz = _timeline_columns(d, [start, end, inner_start, inner_end])
    if sort:
        d = d.sort_values(start, kind="stable")
    names = d[label].astype(str).tolist()
    fig, ax = get_ax(ax, figsize=(10, 0.5 * len(d) + 1.5))
    _timeline(ax, names, d[start], d[end], [colors[0]] * len(d), 0.7, zorder=2)
    _timeline(ax, names, d[inner_start], d[inner_end], [colors[1]] * len(d), 0.35, zorder=3)
    ax.set_yticks(np.arange(len(d))[::-1], names)
    ax.xaxis_date()
    _date_axis(ax, tz=tz)
    ax.legend(handles=[Patch(color=colors[0], label=outer_name), Patch(color=colors[1], label=inner_name)],
              frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    table = d[[label, start, end, inner_start, inner_end]].reset_index(drop=True)
    table["outer_days"] = (table[end] - table[start]).dt.total_seconds() / 86400
    table["inner_days"] = (table[inner_end] - table[inner_start]).dt.total_seconds() / 86400
    table["inner_share"] = table["inner_days"] / table["outer_days"].replace(0, np.nan)
    return VizResult(fig, ax, table)


def animated_bubble(
    data: pd.DataFrame,
    time: str,
    x: str,
    y: str,
    size: str,
    color: str | None = None,
    label: str | None = None,
    max_area: float = 1500.0,
    interval: int = 400,
    colors: Sequence[str] | str | None = None,
) -> VizResult:
    """Animated bubble chart ("Gapminder" style) over the distinct values of *time*.

    Two choices keep frames comparable: bubble **area** is proportional to
    *size* using one scale for the whole animation (not rescaled per frame),
    and the axes limits are fixed across frames. Colours map to *color*
    categories consistently.

    Returns a :class:`VizResult` whose ``info["animation"]`` is a Matplotlib
    ``FuncAnimation``. Save it with ``anim.save("out.gif")`` or show it in a
    notebook with ``IPython.display.HTML(anim.to_jshtml())``.
    """
    check_dataframe(data, [time, x, y, size, color, label])
    check_numeric(data, x, y, size)
    if data[x].notna().sum() == 0 or data[y].notna().sum() == 0:
        raise ValueError("x and y need at least one non-missing value")
    data = data.copy()
    for c in dict.fromkeys((x, y, size)):  # nullable (Int64/Float64) columns -> float with NaN
        data[c] = data[c].to_numpy(dtype=float, na_value=np.nan)
    if (data[size] < 0).any():
        raise ValueError("size must be non-negative")
    if data[size].notna().sum() == 0:
        raise ValueError(f"{size!r} has no non-missing values, so no bubble can be drawn")
    frames = sorted(pd.unique(data[time].dropna()))
    cats = category_order(data[color]) if color is not None else [None]
    cols = palette(len(cats), colors)
    if color is not None and data[color].isna().any():  # rows without a colour category are drawn grey, not dropped
        cats, cols = cats + [None], cols + [NEUTRAL]
    smax = float(data[size].max()) or 1.0
    fig, ax = plt.subplots(figsize=(9, 6))
    # Missing values are skipped (pandas min/max ignore NaN) so the fixed limits stay finite.
    pad_x = 0.08 * ((data[x].max() - data[x].min()) or 1)
    pad_y = 0.08 * ((data[y].max() - data[y].min()) or 1)
    xlim = (data[x].min() - pad_x, data[x].max() + pad_x)
    ylim = (data[y].min() - pad_y, data[y].max() + pad_y)

    def draw(i: int):
        ax.clear()
        cur = data[data[time] == frames[i]]
        for c, col in zip(cats, cols):
            if color is None:
                sub, name = cur, None
            elif c is None:
                sub, name = cur[cur[color].isna()], "(missing)"
            else:
                sub, name = cur[cur[color] == c], str(c)
            ax.scatter(sub[x], sub[y], s=sub[size] / smax * max_area, color=col, alpha=0.65, edgecolors="white",
                       label=name)
            if label is not None:  # label only the bubbles that are drawn, and only when there is a label
                for _, r in sub.dropna(subset=[x, y, size, label]).iterrows():
                    ax.annotate(str(r[label]), (r[x], r[y]), fontsize=7, ha="center", va="center")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        stamp = frames[i]
        stamp = stamp.strftime("%Y-%m-%d") if hasattr(stamp, "strftime") else str(stamp)
        ax.text(0.98, 0.04, stamp, transform=ax.transAxes, ha="right", fontsize=22, color=NEUTRAL, alpha=0.5)
        if color is not None:
            ax.legend(title=str(color), frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.spines[["top", "right"]].set_visible(False)
        return ax.collections

    draw(0)
    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=len(frames), interval=interval, repeat=False)
    table = data.groupby(time)[[x, y, size]].mean().reset_index()
    return VizResult(fig, ax, table, {"animation": anim, "frames": frames, "area_scale": max_area / smax})
