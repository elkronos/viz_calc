"""Shared plumbing: the result container, input validation and palettes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgb
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
        the returned path includes it. File-like objects are passed through.
        """
        if isinstance(self.figure, list):
            from pathlib import Path

            p = Path(path)
            return [VizResult(f).save(str(p.with_name(f"{p.stem}_{i}{p.suffix}")), **kwargs)
                    for i, f in enumerate(self.figure, start=1)]
        import os
        from pathlib import Path

        is_path = isinstance(path, (str, os.PathLike))
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
            else:  # text streams get str, binary ones (BytesIO, open(..., "wb")) get UTF-8 bytes
                import io

                html = self.figure.to_html(**kwargs)
                path.write(html if isinstance(path, io.TextIOBase) else html.encode("utf-8"))
        else:
            self.figure.write_image(path, format=fmt, **kwargs)
        return str(path) if is_path else path

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
    colormap; a sequence is cycled.
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
    return [colors[i % len(colors)] for i in range(n)]


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
    """
    present = pd.unique(values.dropna())
    if len(present) == 0:
        raise ValueError(f"column {values.name!r} has no non-missing values")
    if order is not None:
        order = list(order)
        duplicated = sorted({str(o) for o in order if order.count(o) > 1})
        if duplicated:
            raise ValueError(f"order for {values.name!r} repeats level(s): {duplicated}")
        unknown = [o for o in order if o not in set(present)]
        if unknown and not allow_absent:
            raise ValueError(f"order for {values.name!r} contains levels not in the data: {unknown}")
        if complete:
            left_out = [p for p in present if p not in set(order)]
            if left_out:
                hint = f"; listed but not in the data (a typo?): {unknown}" if unknown else ""
                raise ValueError(f"order for {values.name!r} must list every level; missing: {left_out}{hint}")
        return order
    if isinstance(values.dtype, pd.CategoricalDtype):
        return [c for c in values.cat.categories if c in set(present)]
    try:
        return sorted(present)
    except TypeError:
        return list(present)


_SUFFIXES = ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K"))


def _strip(text: str) -> str:
    return text.rstrip("0").rstrip(".") if "." in text else text


def abbreviate(num: float, digits: int = 1) -> str:
    """Compact number label. Keeps the sign.

    * 10,000 and above: K/M/B/T suffix with *digits* decimals (``12.3K``).
    * 1 to 9,999: thousands separator, up to *digits* decimals (``1,105``).
    * Below 1: *digits* + 1 significant digits (``0.034``, ``0.5``), so
      small rates and proportions never collapse to ``0``.
    """
    if num is None or pd.isna(num):
        return "NA"
    sign = "-" if num < 0 else ""
    num = abs(float(num))
    if num == 0:
        return "0"
    for threshold, suffix in _SUFFIXES:
        if num >= max(threshold, 1e4):
            return f"{sign}{_strip(f'{num / threshold:.{digits}f}')}{suffix}"
    if num < 1:
        return f"{sign}{float(f'{num:.{digits + 1}g}'):g}"
    return f"{sign}{_strip(f'{num:,.{digits}f}')}"


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
