"""Shared plumbing: the result container, input validation and palettes."""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from typing import Any, ParamSpec, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.category import StrCategoryFormatter
from matplotlib.colors import is_color_like, to_hex, to_rgb, to_rgba
from matplotlib.figure import FigureBase
from matplotlib.text import Text
from matplotlib.ticker import FixedFormatter, FuncFormatter, ScalarFormatter
from matplotlib.transforms import Bbox

# Okabe & Ito (2008) "Color Universal Design" palette. The eight colours stay
# distinguishable under the common forms of colour-vision deficiency.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
)

#: Neutral grey used for context marks (reference lines, raw data, gridlines).
NEUTRAL = "#7f7f7f"

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class VizResult:
    """What every plotting function returns.

    A *visual calculator* gives you both the picture and the numbers behind it,
    so the figure can be reported and the statistics can be cited or tested.

    Attributes
    ----------
    figure
        The Matplotlib ``Figure`` (or a Plotly ``Figure`` for the interactive
        functions).
    axes
        The main ``Axes`` or an array of them. ``None`` for Plotly output.
    table
        A tidy ``DataFrame`` holding the values that were drawn.
    info
        Extra scalar results (test statistics, parameters used, notes).

    A result can be pickled, figure included (for a cache or a process
    pool), with two exceptions: ``animated_bubble``'s ``info["animation"]``,
    a Matplotlib ``FuncAnimation``, which never pickles, and the figure of
    ``network_map(directed=True)`` with networkx older than 3.3.
    """

    figure: Any
    axes: Any = None
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    info: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str, **kwargs: Any) -> str | list[str]:
        """Save the figure to *path* (PNG/SVG/PDF for Matplotlib, HTML for Plotly).

        Multi-page results (a list of figures) are saved as ``name_1.png``,
        ``name_2.png``, … and the list of paths is returned. A path without an
        extension gets Matplotlib's default format (``.html`` for Plotly), and
        the returned path includes it. File-like objects are passed through;
        a multi-page result can be written to one only as a multi-page PDF
        (``format="pdf"``).
        """
        import os
        from pathlib import Path

        is_path = isinstance(path, (str, os.PathLike))
        if isinstance(self.figure, list):
            if not is_path:
                return self._save_pages_pdf(path, **kwargs)
            p = Path(path)
            return [VizResult(f).save(str(p.with_name(f"{p.stem}_{i}{p.suffix}")), **kwargs)
                    for i, f in enumerate(self.figure, start=1)]
        if hasattr(self.figure, "savefig"):
            if is_path and not Path(path).suffix and "format" not in kwargs:  # Matplotlib would add one
                path = f"{path}.{plt.rcParams['savefig.format']}"
            kwargs.setdefault("bbox_inches", "tight")
            kwargs.setdefault("dpi", 150)
            self.figure.savefig(path, **kwargs)
            return str(path) if is_path else path
        fmt = kwargs.pop("format", None) or (Path(path).suffix.lstrip(".").lower() if is_path else "") or "html"
        if is_path and not Path(path).suffix:
            path = f"{path}.{fmt}"
        if fmt in ("html", "htm"):
            if is_path:
                self.figure.write_html(path, **kwargs)
            else:  # text streams take str; binary ones (BytesIO, open(..., "wb")) reject it, so send UTF-8 bytes
                html = self.figure.to_html(**kwargs)
                try:
                    path.write(html)
                except TypeError:
                    path.write(html.encode("utf-8"))
        else:
            self.figure.write_image(path, format=fmt, **kwargs)
        return str(path) if is_path else path

    def _save_pages_pdf(self, stream: Any, **kwargs: Any) -> Any:
        """Write every page of a multi-page result into one PDF on a file-like *stream*."""
        fmt = str(kwargs.pop("format", None) or plt.rcParams["savefig.format"]).lower()
        if fmt != "pdf":
            raise ValueError("a multi-page result needs a file path (pages are saved as name_1.png, name_2.png, ...); "
                             "to write it to a file-like object pass format='pdf', which puts every page in one PDF")
        from matplotlib.backends.backend_pdf import PdfPages

        kwargs.setdefault("bbox_inches", "tight")
        kwargs.setdefault("dpi", 150)
        with PdfPages(stream) as pdf:
            for page in self.figure:
                pdf.savefig(page, **kwargs)
        return stream

    def _repr_html_(self) -> str | None:  # pragma: no cover - notebook display
        if hasattr(self.figure, "_repr_html_"):
            return self.figure._repr_html_()
        return None


def check_dataframe(data: Any, columns: Iterable[str | None] = ()) -> pd.DataFrame:
    """Raise a clear error unless *data* is a DataFrame containing *columns*."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    missing = [c for c in columns if c is not None and c not in data.columns]
    if missing:
        raise KeyError(f"column(s) not found in data: {missing}. Available: {list(data.columns)}")
    if data.empty:
        raise ValueError("data has no rows")
    return data


def check_numeric(data: pd.DataFrame, *columns: str) -> None:
    """Raise unless every column in *columns* has a numeric dtype."""
    bad = [c for c in columns if not pd.api.types.is_numeric_dtype(data[c])]
    if bad:
        raise TypeError(f"column(s) must be numeric: {bad}")


def check_finite(data: pd.DataFrame, *columns: str) -> None:
    """Raise if a numeric column in *columns* holds ``inf`` or ``-inf``; missing values are allowed.

    An infinite value would otherwise reach a sum, a mean or an axis limit
    and give a wrong number or fail later with a message that does not name
    the column.
    """
    def infinite(s: pd.Series) -> bool:
        if pd.api.types.is_complex_dtype(s):
            return bool(np.isinf(s.to_numpy()).any())
        if not pd.api.types.is_float_dtype(s):  # integers and booleans cannot be infinite
            return False
        return bool(np.isinf(s.to_numpy(dtype=float, na_value=np.nan)).any())

    bad = [c for c in dict.fromkeys(columns) if infinite(data[c])]
    if bad:
        raise ValueError(f"column(s) hold infinite values (inf or -inf): {bad}; drop or replace those rows first")


def check_choice(name: str, value: Any, choices: Sequence[Any]) -> None:
    """Raise unless *value* is one of *choices*."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")


def check_count(name: str, value: Any, minimum: int = 1) -> None:
    """Raise unless *value* is an ``int`` of at least *minimum* (a layout count such as ``ncols``).

    A whole-number float such as ``2.0`` is rejected too, with a message that
    asks for the ``int``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        if isinstance(value, (float, np.floating)) and float(value).is_integer() and value >= minimum:
            raise ValueError(f"{name} must be a whole number of at least {minimum} given as an int, "
                             f"got the float {value!r}; use {int(value)}")
        raise ValueError(f"{name} must be a whole number of at least {minimum}, got {value!r}")


def check_range(name: str, value: Any, low: float, high: float, include_high: bool = True) -> float:
    """*value* as a float, after checking it is a real number from *low* to *high* (inclusive unless told otherwise).

    Text, booleans and missing values are rejected.
    """
    number = math.nan
    if not isinstance(value, (bool, np.bool_, str, bytes, complex)) and np.ndim(value) == 0:
        try:
            number = float(value)
        except (TypeError, ValueError):
            pass
    if not (low <= number <= high) or (number == high and not include_high):
        interval = f"[{low:g}, {high:g}{']' if include_high else ')'}"
        raise ValueError(f"{name} must be a number in {interval}, got {value!r}")
    return number


def check_has_values(data: pd.DataFrame, *columns: str | None) -> None:
    """Raise unless every column in *columns* has at least one non-missing value."""
    for c in columns:
        if c is not None and not data[c].notna().any():
            raise ValueError(f"column {c!r} has no non-missing values")


def check_not_reserved(reserved: Sequence[str], **roles: Any) -> None:
    """Raise if a column given for one of *roles* is named like one of the table's own columns, *reserved*.

    The output column would otherwise overwrite that column's values in the
    returned table. Roles given as ``None`` are ignored.
    """
    for role, column in roles.items():
        if isinstance(column, str) and column in reserved:
            raise ValueError(f"rename the {role} column {column!r}: the table uses {list(reserved)} "
                             "for its own columns")


def check_distinct(**roles: str | None) -> None:
    """Raise unless the columns given for different roles, such as ``start`` and ``end``, are different columns.

    Roles given as ``None`` are ignored.
    """
    seen: dict[Any, str] = {}
    for role, column in roles.items():
        if column is None:
            continue
        if column in seen:
            raise ValueError(f"{seen[column]} and {role} must be different columns; both are {column!r}")
        seen[column] = role


def column_list(name: str, columns: Any, distinct: bool = False) -> list[Any]:
    """*columns* (any iterable of column names) as a list.

    A lone string is rejected rather than split into its characters, which
    could silently pick single-letter columns. With ``distinct=True`` a
    repeated column is rejected too, rather than used twice.
    """
    if isinstance(columns, str):
        raise TypeError(f"{name} must be a list of column names, not a str; use [{columns!r}] for one column")
    try:
        columns = list(columns)
    except TypeError:
        raise TypeError(f"{name} must be a list of column names, got {type(columns).__name__}") from None
    repeated = [c for i, c in enumerate(columns) if c in columns[:i]] if distinct else []
    if repeated:
        raise ValueError(f"{name} repeats column(s): {list(dict.fromkeys(repeated))}")
    return columns


def select_columns(data: pd.DataFrame, columns: Iterable[Any]) -> pd.DataFrame:
    """The *columns* of *data*, in the order given.

    ``data[columns]`` and ``data.loc[:, columns]`` read a list of booleans as
    a mask, so columns labelled ``False`` and ``True`` (as a pivot on a
    boolean column gives) would not be selected; this looks them up by label.
    """
    columns = list(columns)
    if data.columns.is_unique:
        positions = data.columns.get_indexer(columns)
        if (positions >= 0).all():
            return data.iloc[:, positions]
    return data[columns]


#: Tick formatters that show text from the data, such as the labels given to ``set_xticks``.
_LABEL_FORMATTERS = (FixedFormatter, FuncFormatter, StrCategoryFormatter)

#: Matplotlib reads text with a pair of unescaped ``$`` as mathtext; text made under this setting is drawn as is.
_NO_MATH = {"text.parse_math": False}


def _artists(artist: Any) -> Iterator[Any]:
    """*artist* and the artists under it; of an axis, only the axis label (not its ticks or offset text)."""
    yield artist
    for child in [artist.label] if isinstance(artist, Axis) else artist.get_children():
        yield from _artists(child)


class _Text:
    """The text on the caller's Axes before a plotting call, so that the text the call writes can be told apart."""

    def __init__(self, axes: Iterable[Axes]) -> None:
        self.figures = list(dict.fromkeys(ax.figure for ax in axes))
        found = [a for figure in self.figures for a in _artists(figure)]
        self.texts = {t: t.get_text() for t in found if isinstance(t, Text)}
        self.formatters = {a: (a.get_major_formatter(), a.get_minor_formatter()) for a in found if isinstance(a, Axis)}
        self.parse = {a: (a.majorTicks[0].label1.get_parse_math(), a.minorTicks[0].label1.get_parse_math())
                      for a in self.formatters}

    def settle(self, figures: Iterable[Any]) -> None:
        """Draw the text the call wrote in *figures* as written, and leave the rest to Matplotlib's own rules.

        Text the call made or changed is drawn as written if it holds a
        ``$``, and so are the tick labels of an axis the call gave such
        labels to. The tick labels of an axis whose formatter the call left
        alone keep the setting they had before the call, so those an earlier
        call wrote on the same figure stay as written. All other text
        follows ``text.parse_math``, so a title the caller sets later can
        still use mathtext, and so can the numbers of a log axis or the
        ticks Matplotlib adds when it draws.
        """
        default = plt.rcParams["text.parse_math"]
        for figure in dict.fromkeys([*figures, *self.figures]):
            for artist in _artists(figure):
                if isinstance(artist, Text) and self.texts.get(artist) != artist.get_text():
                    artist.set_parse_math(default and "$" not in artist.get_text())
                elif isinstance(artist, Axis):
                    before = self.formatters.get(artist, (None, None))
                    kept = self.parse.get(artist, (None, None))
                    for ticks, formatter, old, parse_before, locs in (
                        (artist.majorTicks, artist.get_major_formatter(), before[0], kept[0], artist.get_majorticklocs),
                        (artist.minorTicks, artist.get_minor_formatter(), before[1], kept[1], artist.get_minorticklocs),
                    ):
                        if formatter is old:
                            parse = parse_before
                        else:
                            written = isinstance(formatter, _LABEL_FORMATTERS)
                            parse = default and not (written and any("$" in str(t) for t in formatter.format_ticks(locs())))
                        for tick in ticks:
                            tick.label1.set_parse_math(parse)
                            tick.label2.set_parse_math(parse)
                    artist.offsetText.set_parse_math(default)


@contextlib.contextmanager
def text_as_written(*axes: Axes) -> Iterator[None]:
    """Draw the text made in the block, and the text it writes on *axes*, as written: a ``$`` never starts mathtext.

    The plotting functions get this from :func:`cleanup_on_error`; it also
    decorates callbacks that draw after the call has returned, such as
    animation frames.
    """
    text = _Text(axes)
    with plt.rc_context(_NO_MATH):
        yield
    text.settle([])


def _figures(result: Any) -> list[Any]:
    """The Matplotlib figures in a plotting function's *result* (none for a Plotly figure)."""
    figure = result.figure if isinstance(result, VizResult) else None
    return [f for f in (figure if isinstance(figure, list) else [figure]) if isinstance(f, FigureBase)]


def cleanup_on_error(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator for the plotting functions: their text is drawn as written, and a failed call closes its figures.

    Category values, column names and labels are drawn as written: a pair
    of ``$`` in them never starts Matplotlib mathtext, so ``$5-$10`` keeps
    its dollar signs and ``$a^$`` cannot fail when the figure is drawn,
    during the call or when it is saved later. This holds for text the call
    writes on a caller's *ax* too; other text there is left alone.
    ``text.parse_math`` is off only while the call runs.

    A failed call would otherwise leave a half-drawn figure registered with
    pyplot, which Jupyter or ``plt.show()`` would then display. Figures that
    existed before the call (such as the one holding a caller's *ax*) are
    left open.
    """
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        before = set(plt.get_fignums())
        text = _Text(a for a in (*args, *kwargs.values()) if isinstance(a, Axes))
        try:
            with plt.rc_context(_NO_MATH):
                result = func(*args, **kwargs)
        except BaseException:
            for num in set(plt.get_fignums()) - before:
                plt.close(num)
            raise
        text.settle(_figures(result))
        return result

    return wrapper


def get_ax(ax: Axes | None = None, figsize: tuple[float, float] = (8, 5), **subplot_kw: Any) -> tuple[Any, Axes]:
    """Return ``(figure, axes)``, creating a new figure only if *ax* is ``None``."""
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=subplot_kw or None)
    return fig, ax


def palette(n: int, colors: Sequence[str] | str | None = None) -> list[str]:
    """Return *n* colours.

    ``None`` uses Okabe–Ito for up to eight categories and the perceptually
    uniform ``viridis`` colormap beyond that. The eighth Okabe–Ito colour is
    black, so under a style with a dark ``rcParams["axes.facecolor"]`` (such
    as ``dark_background``) it is white instead. A string names any
    Matplotlib colormap; a sequence of colours is cycled.
    """
    if n <= 0:
        return []
    if colors is None:
        if n <= len(OKABE_ITO):
            return [foreground(None, c) if c == OKABE_ITO[7] else c for c in OKABE_ITO[:n]]
        colors = "viridis"
    if isinstance(colors, str):
        cmap = plt.get_cmap(colors)
        return [to_hex(c) for c in cmap(np.linspace(0, 1, n) if n > 1 else [0.5])]
    colors = list(colors)
    if not colors:
        raise ValueError("colors must not be empty")
    bad = [c for c in colors if not is_color_like(c)]
    if bad:
        raise ValueError(f"colors contains invalid colour(s): {bad}")
    return [colors[i % len(colors)] for i in range(n)]


def slot_colors(colors: Any, slots: Sequence[str], name: str = "colors") -> list[Any]:
    """Validate colours given for fixed roles, such as ``(below, above)``, and return them as a list.

    A colormap name is rejected: its letters would otherwise be read as one
    colour each (``"viridis"`` gives red for the third role).
    """
    layout = f"{len(slots)} colours ({', '.join(slots)})"
    if isinstance(colors, str) or not isinstance(colors, Iterable):
        raise ValueError(f"{name} must be a tuple of {layout}, got {colors!r}")
    colors = list(colors)
    if len(colors) != len(slots):
        raise ValueError(f"{name} must be a tuple of {layout}, got {len(colors)}: {colors!r}")
    bad = [c for c in colors if not is_color_like(c)]
    if bad:
        raise ValueError(f"{name} contains invalid colour(s): {bad}")
    return colors


class _VisibleBar:
    """The visible part of a bar, in display units: the box ``Axes.bar_label(label_type="center")`` centres on.

    Matplotlib computes it with a lambda, which pickle cannot store; an
    instance does the same and pickles with its figure.
    """

    def __init__(self, bar: Any) -> None:
        self.bar = bar

    def __call__(self, renderer: Any) -> Bbox:
        return Bbox.intersection(self.bar.get_window_extent(renderer), self.bar.get_clip_box()) or Bbox.null()


def center_labels(ax: Axes, bars: Any, labels: Sequence[str], **kwargs: Any) -> list[Any]:
    """``ax.bar_label(bars, labels, label_type="center", **kwargs)``, with labels that can be pickled."""
    texts = ax.bar_label(bars, labels=labels, label_type="center", **kwargs)
    for text, bar in zip(texts, bars):
        text.xycoords = _VisibleBar(bar)
    return texts


def text_color(background: str) -> str:
    """Black or white, whichever reads better on *background* (WCAG relative luminance)."""
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in to_rgb(background))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    # Pick the colour with the higher WCAG contrast ratio.
    return "black" if (lum + 0.05) / 0.05 >= 1.05 / (lum + 0.05) else "white"


def foreground(ax: Axes | None = None, color: str = "black") -> str:
    """*color* on a light face, its mirror image (one minus each channel) on a dark one.

    *color* is a grey chosen for a white face: ``"black"`` for reference lines,
    error bars and labels drawn on the axes, a light grey for gridlines. The
    face is read when the chart function runs, from *ax* (``rcParams["axes.facecolor"]``
    without one), so the default style and other light styles keep *color* and
    a dark style such as ``dark_background`` gets white marks and a dark grid.
    A transparent face falls back to the figure's, and a transparent figure to
    ``rcParams["text.color"]`` (light text means a dark background).
    """
    if ax is None:
        faces = [plt.rcParams["axes.facecolor"], plt.rcParams["figure.facecolor"]]
    else:
        faces = [ax.get_facecolor(), ax.figure.get_facecolor(), ax.figure.figure.get_facecolor()]
    for face in faces:
        rgba = to_rgba(face)
        if rgba[3] > 0:
            dark = text_color(to_hex(rgba[:3])) == "white"
            break
    else:
        dark = text_color(plt.rcParams["text.color"]) == "black"
    return to_hex([1 - c for c in to_rgb(color)]) if dark else color


def category_order(values: pd.Series, order: Sequence[Any] | None = None, complete: bool = False,
                   allow_absent: bool = False) -> list[Any]:
    """Levels of a categorical series in a stable order.

    Uses *order* if given (every entry must exist), the categorical order for
    ``Categorical`` data, and sorted order otherwise. With ``complete=True``
    the *order* must also list every level present, for charts whose shares
    would otherwise be silently renormalized over a subset. With
    ``allow_absent=True`` the *order* may name levels that do not occur (for
    example the unused points of a response scale); they are drawn as zero.

    Levels taken from the data come back as pandas or Python scalars (a
    ``Timestamp``, not a ``numpy.datetime64``). An *order* is returned as
    given and matched to the data by value, so it may hold either kind.
    """
    if isinstance(order, str):
        raise TypeError(f"order for {values.name!r} must be a list of levels, not a str")
    present = values.dropna().drop_duplicates().tolist()
    if not present:
        raise ValueError(f"column {values.name!r} has no non-missing values")
    known = {_level_key(p) for p in present}
    if order is not None:
        order = list(order)
        duplicated = sorted({str(o) for o in order if order.count(o) > 1})
        if duplicated:
            raise ValueError(f"order for {values.name!r} repeats level(s): {duplicated}")
        unknown = [o for o in order if _level_key(o) not in known]
        if unknown and not allow_absent:
            raise ValueError(f"order for {values.name!r} contains levels not in the data: {unknown}")
        if complete:
            listed = {_level_key(o) for o in order}
            left_out = [p for p in present if _level_key(p) not in listed]
            if left_out:
                hint = f"; listed but not in the data (a typo?): {unknown}" if unknown else ""
                raise ValueError(f"order for {values.name!r} must list every level; missing: {left_out}{hint}")
        return order
    if isinstance(values.dtype, pd.CategoricalDtype):
        return [c for c in values.cat.categories if _level_key(c) in known]
    try:
        return sorted(present)
    except TypeError:
        return present


def level_mask(values: pd.Series, level: Any) -> pd.Series:
    """``values == level``: which rows of *values* hold *level*, also for a column backed by a read-only array.

    pandas before 2.2 cannot compare an object column whose array is
    read-only (a frame built with ``copy=False`` over a read-only NumPy
    array, for example) with a scalar ("buffer source array is read-only"),
    so such a column is compared through a copy. ``isin`` is not used
    because it misses ``numpy.datetime64`` values held in an object column.
    """
    try:
        return values == level
    except ValueError as exc:
        if values.dtype != object or "read-only" not in str(exc):
            raise
        return values.copy() == level


def with_missing(levels: Sequence[Any], values: pd.Series) -> list[Any]:
    """*levels*, then ``None`` if *values* has missing values: the level of the rows whose group is missing.

    :func:`group_mask` selects those rows and :func:`group_label` names the
    level "(missing)".
    """
    return [*levels, None] if values.isna().any() else list(levels)


def group_mask(values: pd.Series, level: Any) -> pd.Series:
    """:func:`level_mask`, except that the level ``None`` selects the missing values."""
    return values.isna() if level is None else level_mask(values, level)


def group_label(level: Any) -> str:
    """:func:`level_label`, except that the level ``None`` reads "(missing)"."""
    return "(missing)" if level is None else level_label(level)


def group_key(values: pd.Series) -> pd.Series:
    """*values* as a key for ``groupby(dropna=False)``, which makes its missing values a group of their own.

    pandas before 3.0 gets that wrong for a dictionary-encoded Arrow column,
    which ``read_parquet(dtype_backend="pyarrow")`` returns for a
    Categorical: it drops the missing values, or adds them to the last level
    when sorting. Such a column is grouped by its decoded values.
    """
    dtype = values.dtype
    if isinstance(dtype, pd.ArrowDtype) and dtype.type is pd.CategoricalDtype.type:  # a dictionary-encoded column
        return values.astype(pd.ArrowDtype(dtype.pyarrow_dtype.value_type))
    return values


def level_codes(levels: Sequence[Any], values: pd.Series) -> np.ndarray:
    """The position of each of *values* in *levels*, -1 where a value is missing or not one of them.

    pandas before 3.0 finds no value of an Arrow-backed timestamp column in
    an Index of Timestamps, and pandas 2.2 none of an Arrow-backed duration
    column, so an Arrow-backed column is matched as objects.
    """
    if isinstance(values.dtype, pd.ArrowDtype):
        values = values.astype(object)
    return pd.Index(levels).get_indexer(values)


def _level_key(value: Any) -> Any:
    # On NumPy 1.x a numpy.datetime64 (or timedelta64) hashes differently from the equal pandas Timestamp
    # (Timedelta), so set lookups would miss; compare levels in their pandas form.
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)
    if isinstance(value, np.timedelta64):
        return pd.Timedelta(value)
    return value


def level_label(value: Any) -> str:
    """Text for a category level. A date with no time of day reads ``2024-01-01``, with or without a time zone."""
    value = _level_key(value)
    if isinstance(value, datetime) and not isinstance(value, pd.Timestamp):
        value = pd.Timestamp(value)
    if isinstance(value, pd.Timestamp):
        wall = value.tz_localize(None)  # local clock time; normalizing in the zone could land on a skipped midnight
        if wall == wall.normalize():
            return value.strftime("%Y-%m-%d")
    return str(value)


def level_percent(level: Any) -> str:
    """A confidence or coverage *level* as a percentage with every digit it has: ``0.975`` reads ``97.5%``.

    Rounding to whole percent would label a 97.5% interval as 98% and a
    99.9% one as an impossible 100%.
    """
    # The shortest text that reads back as the level; a NumPy float gives it for its own precision (float32 0.95).
    text = str(level) if isinstance(level, np.floating) else repr(float(level))
    return f"{(Decimal(text) * 100).normalize():f}%"


_SUFFIXES = ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K"))


def _strip(text: str) -> str:
    return text.rstrip("0").rstrip(".") if "." in text else text


def abbreviate(num: float, digits: int = 1) -> str:
    """Compact number label. Keeps the sign.

    * 10,000 and above: K/M/B/T suffix with *digits* decimals (``12.3K``).
    * 1 to 9,999: thousands separator, up to *digits* decimals (``1,105``).
    * Below 1: *digits* + 1 significant digits (``0.034``, ``0.5``), so
      small rates and proportions never collapse to ``0``.
    * Beyond the T range (a rounded 1,000T and above): scientific notation
      with *digits* decimals (``2.5e18``).
    * Infinite values: ``inf`` and ``-inf``.

    The unit is chosen after rounding, so 999,999 reads ``1M`` (not
    ``1000K``) and 9,999.96 reads ``10K``.
    """
    if num is None or pd.isna(num):
        return "NA"
    sign = "-" if num < 0 else ""
    num = abs(float(num))
    if num == 0:
        return "0"
    if math.isinf(num):
        return f"{sign}inf"
    if num < 1:  # formatted once: reading the rounded text back as a float re-expands a subnormal to six digits
        return f"{sign}{num:.{digits + 1}g}"
    if float(f"{num:.{digits}f}") < 1e4:
        return f"{sign}{_strip(f'{num:,.{digits}f}')}"
    # The first of K, M, B, T in which the rounded value stays below 1000.
    found = next(((t, s) for t, s in reversed(_SUFFIXES) if float(f"{num / t:.{digits}f}") < 1000), None)
    if found is None:  # a count of T would print every digit
        mantissa, exponent = f"{num:.{digits}e}".split("e")
        return f"{sign}{_strip(mantissa)}e{int(exponent)}"
    unit, suffix = found
    return f"{sign}{_strip(f'{num / unit:.{digits}f}')}{suffix}"


def exact_mean(values: Any) -> float:
    """Mean of *values*, correctly rounded (missing values are not skipped).

    Floating-point summation rounds at every step and ``sum / n`` rounds
    again, which can put a value that sits exactly on the mean just below
    it. Here the sum is found exactly and the division is exact, so the
    result is rounded only once.
    """
    v = np.asarray(values, dtype=float).ravel().tolist()
    if not v:
        return math.nan
    try:
        parts = [math.fsum(v)]
        # fsum is correctly rounded, so each pass recovers the next piece of what the earlier pieces missed; the
        # remainder shrinks by 2**-53 or more per pass, and the pieces end up adding to the exact sum.
        while math.isfinite(parts[0]):
            rest = math.fsum([*v, *(-p for p in parts)])
            if not rest:
                break
            parts.append(rest)
    except OverflowError:  # the sum leaves the float range, the mean cannot
        return float(sum(map(Fraction, v)) / len(v))
    except ValueError:  # both +inf and -inf
        return math.nan
    if not math.isfinite(parts[0]):
        return parts[0]
    return float(sum(map(Fraction, parts)) / len(v))


def safe_sum(values: pd.Series, by: Any = None, sort: bool = False, min_count: int = 0,
             dropna: bool = True) -> Any:
    """Sum of *values* (per group of *by*) that cannot wrap around, with ``NaN`` rather than ``pd.NA`` for missing.

    Integer sums in int64 wrap around silently, so where an integer sum
    reaches 2**62 (a margin for float rounding) it is taken in float instead.
    Otherwise integer sums stay exact. A result in a nullable dtype (``Int64``,
    ``Float64``) comes back as a NumPy one: int64 when nothing is missing,
    float64 with ``NaN`` otherwise, which Matplotlib can draw. *by* is a
    Series or a list of them. *sort*, *dropna* (``False`` makes missing keys
    a group of their own) and *min_count* are passed on to the pandas
    ``groupby`` and ``sum``.
    """
    def total(v: pd.Series) -> Any:
        if by is None:
            return v.sum(min_count=min_count)
        return v.groupby(by, sort=sort, observed=True, dropna=dropna).sum(min_count=min_count)

    if pd.api.types.is_integer_dtype(values) and not pd.api.types.is_bool_dtype(values):
        floats = total(pd.Series(values.to_numpy(dtype=float, na_value=np.nan), index=values.index))
        if np.any(np.abs(floats) >= 2.0**62):
            return floats
    result = total(values)
    if by is None:
        return np.nan if pd.isna(result) else result
    if isinstance(result.dtype, pd.api.extensions.ExtensionDtype) and pd.api.types.is_numeric_dtype(result):
        dtype = float if result.isna().any() else result.dtype.numpy_dtype
        result = pd.Series(result.to_numpy(dtype=dtype, na_value=np.nan), index=result.index, name=result.name)
    return result


class AbbrevFormatter(ScalarFormatter):
    """Axis formatter: K/M/B/T suffixes when ticks reach 10,000, Matplotlib's defaults otherwise.

    For evenly spaced ticks, one unit is used and the number of decimals is
    the smallest that shows every tick exactly (so a 2,500 step gives
    ``2.5K, 5.0K, 7.5K``); if that needs more than three decimals (a narrow
    range at a large magnitude), Matplotlib's own offset notation is used
    instead. Log-scaled or unevenly spaced ticks are abbreviated one by one.
    With ``absolute=True`` (mirrored axes) no offset is used, so no label or
    offset ever reads as negative.
    """

    def __init__(self, absolute: bool = False) -> None:
        super().__init__()
        self.absolute = absolute
        self._mode: Any = None
        if absolute:
            self.set_useOffset(False)

    def set_locs(self, locs: Sequence[float]) -> None:
        self._mode = self._choose(np.asarray(locs, dtype=float))  # Formatter.locs is deprecated in Matplotlib 3.11
        super().set_locs(locs)

    def _choose(self, ticks: np.ndarray) -> Any:
        ticks = ticks[np.isfinite(ticks)]
        if ticks.size == 0 or np.max(np.abs(ticks)) < 1e4:
            return None
        scale = self.axis.get_scale() if self.axis is not None else "linear"
        steps = np.diff(np.sort(ticks))
        noise = 16 * np.spacing(np.max(np.abs(ticks)))  # float spacing at this magnitude
        if scale != "linear" or (steps.size and not np.allclose(steps, steps[0], rtol=1e-6, atol=noise)):
            labels = [abbreviate(abs(t) if self.absolute else t) for t in ticks]
            return "each" if len(set(labels)) == len(labels) else None  # never repeat a label
        big = float(np.max(np.abs(ticks)))
        if big >= 1e15:  # beyond the T range; Matplotlib's scientific offset reads better than thousands of T
            return None
        unit, suffix = next((t, suf) for t, suf in _SUFFIXES if big >= t)
        values = (np.abs(ticks) if self.absolute else ticks) / unit
        for decimals in range(4):
            labels = [f"{v:.{decimals}f}" for v in values]
            exact = all(abs(float(lab) - v) <= 1e-9 * max(1.0, abs(v)) for lab, v in zip(labels, values))
            if exact and len(set(labels)) == len(set(np.round(values, 12))):
                return unit, suffix, decimals
        return None

    def __call__(self, x: float, pos: int | None = None) -> str:
        value = abs(x) if self.absolute else x
        if self._mode is None:
            text = super().__call__(value, pos)
            return text.lstrip("\u2212-") if self.absolute else text
        if self._mode == "each":
            return abbreviate(value)
        unit, suffix, decimals = self._mode
        return f"{value / unit:.{decimals}f}{suffix}" if value else "0"

    def get_offset(self) -> str:
        return "" if self._mode is not None else super().get_offset()


_DISTRIBUTIONS = {"pptx": "python-pptx"}  # import name -> the package that installs it, where they differ


def require(module: str, extra: str) -> Any:
    """Import an optional dependency or explain how to install it.

    The message names the package to install (plotly, python-pptx), not the module imported
    (plotly.graph_objects, pptx).
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on environment
        top = module.split(".")[0]
        raise ImportError(
            f"This function needs the optional dependency '{_DISTRIBUTIONS.get(top, top)}'. "
            f"Install it with: {install_hint(extra)}"
        ) from exc


def install_hint(extra: str) -> str:
    """The pip command that installs an extra's packages.

    viz_calc is not on PyPI, so ``pip install "viz_calc[extra]"`` would fetch
    whatever someone else publishes under that name. The packages and their
    minimum versions come from the installed package metadata, which is built
    from pyproject.toml; when it has no entry for the extra, the hint installs
    viz_calc with the extra from the git URL.
    """
    import re
    from importlib.metadata import PackageNotFoundError, requires

    try:
        reqs = requires("viz_calc") or []
    except PackageNotFoundError:
        reqs = []
    marker = re.compile(rf"""extra\s*==\s*["']{re.escape(extra)}["']""")
    itself = re.compile(r"viz[-_.]?calc\b", re.IGNORECASE)  # some build tools write all = viz-calc[...]
    packages = [r.split(";")[0].strip() for r in reqs if marker.search(r) and not itself.match(r)]
    if packages:
        return "pip install " + " ".join(f'"{p}"' for p in packages)
    return f'pip install "viz_calc[{extra}] @ git+https://github.com/elkronos/viz_calc"'
