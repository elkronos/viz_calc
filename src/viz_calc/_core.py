"""Shared plumbing: the result container, input validation and palettes."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from typing import Any, ParamSpec, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like, to_hex, to_rgb
from matplotlib.ticker import ScalarFormatter

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


def check_choice(name: str, value: Any, choices: Sequence[Any]) -> None:
    """Raise unless *value* is one of *choices*."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")


def check_count(name: str, value: Any, minimum: int = 1) -> None:
    """Raise unless *value* is a whole number of at least *minimum* (a layout count such as ``ncols``)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be a whole number of at least {minimum}, got {value!r}")


def check_has_values(data: pd.DataFrame, *columns: str | None) -> None:
    """Raise unless every column in *columns* has at least one non-missing value."""
    for c in columns:
        if c is not None and not data[c].notna().any():
            raise ValueError(f"column {c!r} has no non-missing values")


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


def column_list(name: str, columns: Any) -> list[Any]:
    """*columns* (any iterable of column names) as a list.

    A lone string is rejected rather than split into its characters, which
    could silently pick single-letter columns.
    """
    if isinstance(columns, str):
        raise TypeError(f"{name} must be a list of column names, not a str; use [{columns!r}] for one column")
    try:
        return list(columns)
    except TypeError:
        raise TypeError(f"{name} must be a list of column names, got {type(columns).__name__}") from None


def cleanup_on_error(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator: close the Matplotlib figures a plotting call opened if the call raises.

    Otherwise a failed call leaves a half-drawn figure registered with
    pyplot, which Jupyter or ``plt.show()`` would then display. Figures that
    existed before the call (such as the one holding a caller's *ax*) are
    left open.
    """
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        before = set(plt.get_fignums())
        try:
            return func(*args, **kwargs)
        except BaseException:
            for num in set(plt.get_fignums()) - before:
                plt.close(num)
            raise

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
    uniform ``viridis`` colormap beyond that. A string names any Matplotlib
    colormap; a sequence of colours is cycled.
    """
    if n <= 0:
        return []
    if colors is None:
        if n <= len(OKABE_ITO):
            return list(OKABE_ITO[:n])
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


def text_color(background: str) -> str:
    """Black or white, whichever reads better on *background* (WCAG relative luminance)."""
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in to_rgb(background))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    # Pick the colour with the higher WCAG contrast ratio.
    return "black" if (lum + 0.05) / 0.05 >= 1.05 / (lum + 0.05) else "white"


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

    The unit is chosen after rounding, so 999,999 reads ``1M`` (not
    ``1000K``) and 9,999.96 reads ``10K``.
    """
    if num is None or pd.isna(num):
        return "NA"
    sign = "-" if num < 0 else ""
    num = abs(float(num))
    if num == 0:
        return "0"
    if num < 1:
        return f"{sign}{float(f'{num:.{digits + 1}g}'):g}"
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


def require(module: str, extra: str) -> Any:
    """Import an optional dependency or explain how to install it."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            f"This function needs the optional dependency '{module}'. "
            f"Install it with: pip install \"viz_calc[{extra}]\""
        ) from exc
