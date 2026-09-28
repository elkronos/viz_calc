"""viz_calc — visual calculators for pandas data.

Every plotting function returns a :class:`VizResult` holding the figure *and*
the numbers behind it (``.table``, ``.info``), computed with published,
citable methods. Nothing is shown or written to disk unless you ask.

>>> import viz_calc as vc
>>> res = vc.estimation_plot(df, x="treatment", y="score")  # doctest: +SKIP
>>> res.info["comparisons"]                                   # doctest: +SKIP
>>> res.save("effect.png")                                    # doctest: +SKIP
"""

from . import datasets, stats
from ._core import OKABE_ITO, VizResult, palette
from .compare import benchmark_bar, centered_bar, divergent_bar, dumbbell, estimation_plot, likert, lollipop
from .composition import (
                          bullet,
                          circular_bar,
                          donut_grid,
                          funnel,
                          nested_pie,
                          percent_grid,
                          stacked_percentages,
                          upset,
                          waffle,
                          waterfall,
)
from .correlation import compare_correlations, correlogram, quadrant_plot
from .distribution import histogram, raincloud, ridgeplot
from .explore import profile_bars, profile_boxes, profile_scatters, to_pptx
from .multivariate import pca, pca_plot, radar
from .network import network_map, sankey
from .timeseries import animated_bubble, calendar_heatmap, duration_plot, gantt, period_bars, timeseries_fill

__version__ = "1.0.0"

__all__ = [
    "VizResult", "OKABE_ITO", "palette", "stats", "datasets",
    # compare
    "estimation_plot", "benchmark_bar", "lollipop", "dumbbell", "divergent_bar", "centered_bar", "likert",
    # correlation
    "correlogram", "compare_correlations", "quadrant_plot",
    # distribution
    "histogram", "ridgeplot", "raincloud",
    # composition
    "waffle", "percent_grid", "stacked_percentages", "donut_grid", "nested_pie", "circular_bar", "waterfall",
    "funnel", "bullet", "upset",
    # time
    "period_bars", "timeseries_fill", "calendar_heatmap", "gantt", "duration_plot", "animated_bubble",
    # multivariate
    "pca", "pca_plot", "radar",
    # network
    "network_map", "sankey",
    # explore
    "profile_bars", "profile_boxes", "profile_scatters", "to_pptx",
]
