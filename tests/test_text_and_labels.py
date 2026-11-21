"""Text the plotting functions draw: number labels, Matplotlib mathtext, Plotly markup and pickling."""

import io
import math
import pickle

import pandas as pd
import pytest
from matplotlib.container import BarContainer
from matplotlib.transforms import Bbox

import viz_calc as vc
from viz_calc import datasets
from viz_calc._core import abbreviate

# --- abbreviate ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "text"), [(math.inf, "inf"), (-math.inf, "-inf")])
def test_abbreviate_reads_infinity_as_inf(value, text):
    assert abbreviate(value) == text
    assert abbreviate(value, digits=0) == text


@pytest.mark.parametrize(("value", "digits", "text"), [
    (5e-324, 1, "4.9e-324"), (-5e-324, 1, "-4.9e-324"), (5e-324, 0, "5e-324"), (1.23456e-310, 2, "1.23e-310"),
    (0.034, 1, "0.034"), (0.00012345, 1, "0.00012"),
])
def test_abbreviate_gives_digits_plus_one_significant_digits_below_one(value, digits, text):
    assert abbreviate(value, digits=digits) == text


# --- Plotly markup ------------------------------------------------------------------------------------------------

LINK = '<a href="https://example.com/login">Click</a>'
FLOWS = pd.DataFrame({"from": [LINK, "A<br>B", "AT&T"], "to": ["X", "X", "<18"], "n": [3, 2, 1]})


def test_sankey_shows_node_names_as_written():
    pytest.importorskip("plotly")
    res = vc.sankey(FLOWS, source="from", target="to", value="n", title="<b>Flows</b>")
    labels = list(res.figure.data[0].node.label)
    assert labels == ['&lt;a href="https://example.com/login"&gt;Click&lt;/a&gt;', "A&lt;br&gt;B", "AT&amp;T", "X",
                      "&lt;18"]
    assert list(res.table["node"]) == [LINK, "A<br>B", "AT&T", "X", "<18"]  # the table keeps the names
    assert res.figure.layout.title.text == "<b>Flows</b>"  # the caller's own title is passed through


@pytest.mark.parametrize("directed", [False, True])
def test_interactive_network_map_shows_node_names_as_written(directed):
    pytest.importorskip("networkx")
    pytest.importorskip("plotly")
    res = vc.network_map(FLOWS, source="from", target="to", interactive=True, directed=directed)
    points = res.figure.data[-1]
    names = list(res.table["node"])
    escaped = {LINK: '&lt;a href="https://example.com/login"&gt;Click&lt;/a&gt;', "A<br>B": "A&lt;br&gt;B",
               "AT&T": "AT&amp;T", "X": "X", "<18": "&lt;18"}
    assert list(points.text) == [escaped[n] for n in names]
    for name, hover in zip(names, points.hovertext):
        assert hover.startswith(f"<b>{escaped[name]}</b><br>degree ")  # the chart's own markup stays
    assert not any("<a " in t for t in [*points.text, *points.hovertext])


# --- every plotting function --------------------------------------------------------------------------------------

LEVELS = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
BUBBLES = (pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"]),
                         "region": ["north", "south", "north", "south"], "sales": [3.0, 5, 4, 6]}))


def _calls():
    """``name: (function, data, keyword arguments, polar)`` for every Matplotlib plotting function.

    *polar* is ``None`` for a function that takes no *ax*, else whether its *ax* must be polar.
    """
    trial, survey, meas = datasets.trial(), datasets.survey(), datasets.measurements()
    small = pd.DataFrame({"k": list("abcd"), "a": [3.0, 5, 2, 8], "b": [4.0, 1, 6, 7]})
    sales, proj = datasets.sales(), datasets.projects()
    return {
        "estimation_plot": (vc.estimation_plot, trial, dict(x="arm", y="score", n_resamples=200), None),
        "benchmark_bar": (vc.benchmark_bar, trial, dict(x="arm", y="score", threshold=50), False),
        "lollipop": (vc.lollipop, survey, dict(x="team", y="tenure_years"), False),
        "dumbbell": (vc.dumbbell, small, dict(label="k", start="a", end="b", show_values=True), False),
        "divergent_bar": (vc.divergent_bar, small, dict(category="k", left="a", right="b"), False),
        "centered_bar": (vc.centered_bar, trial, dict(x="arm", y="score"), False),
        "likert": (vc.likert, survey, dict(items=["Tools are adequate", "Meetings are useful"], levels=LEVELS), False),
        "correlogram": (vc.correlogram, meas, dict(), False),
        "compare_correlations": (vc.compare_correlations, meas, dict(group="group"), None),
        "quadrant_plot": (vc.quadrant_plot, trial, dict(x="baseline", y="score"), False),
        "histogram": (vc.histogram, trial, dict(x="score", hue="arm"), None),
        "ridgeplot": (vc.ridgeplot, meas, dict(x="depth", group="group"), False),
        "raincloud": (vc.raincloud, trial, dict(x="arm", y="score"), False),
        "waffle": (vc.waffle, survey, dict(category="team"), False),
        "percent_grid": (vc.percent_grid, trial, dict(column="improved", facet="site"), None),
        "stacked_percentages": (vc.stacked_percentages, survey, dict(group="team", category="remote"), False),
        "donut_grid": (vc.donut_grid, small.set_index("k"), dict(), None),
        "nested_pie": (vc.nested_pie, survey, dict(outer="team", inner="remote"), False),
        "circular_bar": (vc.circular_bar, small, dict(label="k", value="a"), True),
        "waterfall": (vc.waterfall, small, dict(label="k", value="a"), False),
        "funnel": (vc.funnel, small.sort_values("a", ascending=False), dict(stage="k", value="a"), False),
        "bullet": (vc.bullet, small.head(1), dict(label="k", value="a", target="b", bands=[2, 5, 9]), False),
        "upset": (vc.upset, pd.DataFrame({"Search": [1, 1, 0, 1], "Export": [0, 1, 1, 1]}).astype(bool), dict(), None),
        "period_bars": (vc.period_bars, sales, dict(date="date", value="online"), False),
        "timeseries_fill": (vc.timeseries_fill, sales.head(60), dict(time="date", series=["online", "store"]), False),
        "calendar_heatmap": (vc.calendar_heatmap, sales, dict(date="date", value="online"), None),
        "gantt": (vc.gantt, proj, dict(task="task", start="start", end="end", group="team"), False),
        "duration_plot": (vc.duration_plot, proj, dict(label="task", start="start", end="end",
                                                       inner_start="work_start", inner_end="work_end"), False),
        "pca_plot": (vc.pca_plot, meas, dict(features=["length", "width", "depth", "mass"], group="group"), False),
        "radar": (vc.radar, meas, dict(metrics=["length", "width", "depth", "mass"], group="group"), True),
        "network_map": (vc.network_map, datasets.network(), dict(source="source", target="target",
                                                                  weight="strength"), False),
        "animated_bubble": (vc.animated_bubble, BUBBLES, dict(time="date", x="sales", y="sales", size="sales",
                                                             color="region", label="region"), None),
        "profile_bars": (vc.profile_bars, meas, dict(), None),
        "profile_boxes": (vc.profile_boxes, meas, dict(), None),
        "profile_scatters": (vc.profile_scatters, meas, dict(), None),
    }


CALLS = _calls()
WITH_AX = [name for name, call in CALLS.items() if call[3] is not None]


def _run(name, data=None, **extra):
    func, frame, kwargs, _ = CALLS[name]
    if name == "network_map":
        pytest.importorskip("networkx")
    return func(frame if data is None else data, **{**kwargs, **extra})


def _figures(res):
    return res.figure if isinstance(res.figure, list) else [res.figure]


# --- pickling -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CALLS))
def test_results_pickle_with_their_figures(name):
    res = _run(name)
    if name == "animated_bubble":  # a FuncAnimation never pickles; its figure and table do
        res = vc.VizResult(res.figure, res.axes, res.table, {k: v for k, v in res.info.items() if k != "animation"})
    back = pickle.loads(pickle.dumps(res))
    pd.testing.assert_frame_equal(back.table, res.table)
    for figure in _figures(back):
        figure.savefig(io.BytesIO(), format="png")


@pytest.mark.parametrize("extra", [{}, {"labels": False}])
@pytest.mark.parametrize("name", ["centered_bar", "likert"])
def test_shares_drawn_both_sides_of_zero_pickle(name, extra):
    back = pickle.loads(pickle.dumps(_run(name, **extra)))
    ticks = back.axes.yaxis if name == "centered_bar" else back.axes.xaxis
    back.figure.canvas.draw()
    assert all("%" in t.get_text() and "-" not in t.get_text() and "\u2212" not in t.get_text()
               for t in ticks.get_ticklabels() if t.get_text())


@pytest.mark.parametrize("extra", [{}, {"horizontal": False}, {"min_label": 101}])
def test_stacked_percentages_pickles(extra):
    back = pickle.loads(pickle.dumps(_run("stacked_percentages", **extra)))
    back.figure.savefig(io.BytesIO(), format="png")


@pytest.mark.parametrize("name", ["centered_bar", "likert", "stacked_percentages"])
def test_centred_bar_labels_stay_on_the_visible_part_of_their_bar_after_pickling(name):
    res = pickle.loads(pickle.dumps(_run(name)))
    ax = res.axes
    if name == "likert":
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, (lo + hi) / 2)  # bars on the right now run past the edge of the axes
    bars = [bar for c in ax.containers if isinstance(c, BarContainer) for bar in c]
    assert len(bars) == len(ax.texts)  # one label per bar, made in the same order
    res.figure.canvas.draw()
    renderer = res.figure.canvas.get_renderer()
    pad = 2 * res.figure.dpi / 72 if name == "centered_bar" else 0  # centered_bar nudges labels 2 points outwards
    checked = 0
    for bar, text in zip(bars, ax.texts):
        visible = Bbox.intersection(bar.get_window_extent(renderer), ax.bbox)
        if not text.get_text() or visible is None:
            continue
        x, y = text.get_window_extent(renderer).get_points().mean(axis=0)
        assert x == pytest.approx((visible.x0 + visible.x1) / 2, abs=1)
        assert y == pytest.approx((visible.y0 + visible.y1) / 2, abs=1 + pad)
        checked += 1
    assert checked >= 4
