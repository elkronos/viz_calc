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
        ``name_2.png``, … and the list of paths is returned.
        """
        if isinstance(self.figure, list):
            from pathlib import Path

            p = Path(path)
            return [VizResult(f).save(str(p.with_name(f"{p.stem}_{i}{p.suffix}")), **kwargs)
                    for i, f in enumerate(self.figure, start=1)]
        if hasattr(self.figure, "savefig"):
            kwargs.setdefault("bbox_inches", "tight")
            kwargs.setdefault("dpi", 150)
            self.figure.savefig(path, **kwargs)
        elif str(path).lower().endswith((".html", ".htm")):
            self.figure.write_html(path, **kwargs)
        else:
            self.figure.write_image(path, **kwargs)
        return str(path)

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


def category_order(values: pd.Series, order: Sequence[Any] | None = None, complete: bool = False) -> list[Any]:
    """Levels of a categorical series in a stable order.

    Uses *order* if given (every entry must exist), the categorical order for
    ``Categorical`` data, and sorted order otherwise. With ``complete=True``
    the *order* must also list every level present, for charts whose shares
    would otherwise be silently renormalized over a subset.
    """
    present = pd.unique(values.dropna())
    if len(present) == 0:
        raise ValueError(f"column {values.name!r} has no non-missing values")
    if order is not None:
        order = list(order)
        unknown = [o for o in order if o not in set(present)]
        if unknown:
            raise ValueError(f"order for {values.name!r} contains levels not in the data: {unknown}")
        if complete:
            left_out = [p for p in present if p not in set(order)]
            if left_out:
                raise ValueError(f"order for {values.name!r} must list every level; missing: {left_out}")
        return order
    if isinstance(values.dtype, pd.CategoricalDtype):
        return [c for c in values.cat.categories if c in set(present)]
    try:
        return sorted(present)
    except TypeError:
        return list(present)


def abbreviate(num: float, digits: int = 1) -> str:
    """Compact number label: exact below 10,000 (``1,105``), then K/M/B/T (``12.3K``). Keeps the sign."""
    if num is None or pd.isna(num):
        return "NA"

    def strip(text: str) -> str:
        return text.rstrip("0").rstrip(".") if "." in text else text

    sign = "-" if num < 0 else ""
    num = abs(float(num))
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e4, "K")):
        if num >= threshold:
            scale = 1e3 if suffix == "K" else threshold
            return f"{sign}{strip(f'{num / scale:.{digits}f}')}{suffix}"
    return f"{sign}{strip(f'{num:,.{digits}f}')}"


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
