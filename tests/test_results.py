"""The result object, saving and export, and how a call treats the caller's Axes and figures."""

from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets

SMALL = pd.DataFrame({"k": list("abcd"), "a": [3.0, 5, 2, 8], "b": [4.0, 1, 6, 7]})
METRICS = ["length", "width", "depth", "mass"]


# --- an Axes in a SubFigure -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("draw, projection", [
    (lambda ax: vc.lollipop(datasets.survey(), x="team", y="tenure_years", ax=ax), None),
    (lambda ax: vc.correlogram(datasets.measurements(), ax=ax), None),
    (lambda ax: vc.bullet(SMALL.head(1), label="k", value="a", target="b", ax=ax), None),
    (lambda ax: vc.radar(datasets.measurements(), metrics=METRICS, group="group", ax=ax), "polar"),
])
def test_result_of_an_axes_in_a_subfigure_holds_the_whole_figure(draw, projection, tmp_path):
    fig = plt.figure(figsize=(10, 4))
    left, right = fig.subfigures(1, 2)
    title = right.suptitle(r"$\alpha$ panel")
    res = draw(left.add_subplot(projection=projection))
    assert res.figure is fig
    buf = BytesIO()
    assert res.save(buf, format="png") is buf and buf.getvalue().startswith(b"\x89PNG")
    assert res.save(str(tmp_path / "chart")).endswith("chart.png")
    assert title.get_parse_math()  # text in the other panel is left alone


def test_to_pptx_takes_a_result_or_a_subfigure(tmp_path):
    pptx = pytest.importorskip("pptx")
    fig = plt.figure()
    left, _ = fig.subfigures(1, 2)
    res = vc.lollipop(datasets.survey(), x="team", y="tenure_years", ax=left.subplots())
    out = vc.to_pptx([res, left], str(tmp_path / "deck.pptx"))
    assert len(pptx.Presentation(out).slides) == 2


def test_a_result_made_from_a_subfigure_keeps_the_whole_figure():
    fig = plt.figure()
    sub = fig.subfigures(1, 2)[0].subfigures(2, 1)[1]  # nested
    assert vc.VizResult(sub).figure is fig
    assert vc.VizResult(fig).figure is fig


# --- what ax= accepts -----------------------------------------------------------------------------------------------

LEVELS = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]


def _ax_calls():
    """``name: (function, data, keyword arguments, polar)`` for every function that takes ``ax=``."""
    trial, survey, meas = datasets.trial(), datasets.survey(), datasets.measurements()
    sales, proj = datasets.sales(), datasets.projects()
    return {
        "benchmark_bar": (vc.benchmark_bar, trial, dict(x="arm", y="score", threshold=50), False),
        "lollipop": (vc.lollipop, survey, dict(x="team", y="tenure_years"), False),
        "dumbbell": (vc.dumbbell, SMALL, dict(label="k", start="a", end="b"), False),
        "divergent_bar": (vc.divergent_bar, SMALL, dict(category="k", left="a", right="b"), False),
        "centered_bar": (vc.centered_bar, trial, dict(x="arm", y="score"), False),
        "likert": (vc.likert, survey, dict(items=["Tools are adequate", "Meetings are useful"], levels=LEVELS), False),
        "correlogram": (vc.correlogram, meas, dict(), False),
        "quadrant_plot": (vc.quadrant_plot, trial, dict(x="baseline", y="score"), False),
        "ridgeplot": (vc.ridgeplot, meas, dict(x="depth", group="group"), False),
        "raincloud": (vc.raincloud, trial, dict(x="arm", y="score"), False),
        "waffle": (vc.waffle, survey, dict(category="team"), False),
        "stacked_percentages": (vc.stacked_percentages, survey, dict(group="team", category="remote"), False),
        "nested_pie": (vc.nested_pie, survey, dict(outer="team", inner="remote"), False),
        "circular_bar": (vc.circular_bar, SMALL, dict(label="k", value="a"), True),
        "waterfall": (vc.waterfall, SMALL, dict(label="k", value="a"), False),
        "funnel": (vc.funnel, SMALL.sort_values("a", ascending=False), dict(stage="k", value="a"), False),
        "bullet": (vc.bullet, SMALL.head(1), dict(label="k", value="a", target="b", bands=[2, 5, 9]), False),
        "period_bars": (vc.period_bars, sales, dict(date="date", value="online"), False),
        "timeseries_fill": (vc.timeseries_fill, sales.head(60), dict(time="date", series=["online", "store"]), False),
        "gantt": (vc.gantt, proj, dict(task="task", start="start", end="end", group="team"), False),
        "duration_plot": (vc.duration_plot, proj, dict(label="task", start="start", end="end",
                                                       inner_start="work_start", inner_end="work_end"), False),
        "pca_plot": (vc.pca_plot, meas, dict(features=METRICS, group="group"), False),
        "radar": (vc.radar, meas, dict(metrics=METRICS, group="group"), True),
        "network_map": (vc.network_map, datasets.network(), dict(source="source", target="target",
                                                                  weight="strength"), False),
    }


AX_CALLS = _ax_calls()


def _ax_call(name):
    if name == "network_map":
        pytest.importorskip("networkx")
    return AX_CALLS[name][:3]


def test_every_function_with_ax_is_listed():
    import inspect

    takes_ax = {n for n in vc.__all__ if callable(getattr(vc, n)) and "ax" in inspect.signature(getattr(vc, n)).parameters}
    assert takes_ax == set(AX_CALLS)


@pytest.mark.parametrize("name", list(AX_CALLS))
def test_ax_of_the_wrong_projection_raises_before_drawing(name):
    func, data, kwargs = _ax_call(name)
    polar = AX_CALLS[name][3]
    fig = plt.figure()
    ax = fig.add_subplot(projection=None if polar else "polar")
    before = len(ax.get_children()), len(fig.get_children())
    with pytest.raises(ValueError, match="polar Axes" if polar else "Cartesian"):
        func(data, ax=ax, **kwargs)
    assert (len(ax.get_children()), len(fig.get_children())) == before


@pytest.mark.parametrize("name", list(AX_CALLS))
def test_ax_must_be_one_axes(name):
    func, data, kwargs = _ax_call(name)
    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"} if AX_CALLS[name][3] else None)
    for wrong in (fig, axes, list(axes), "left"):
        with pytest.raises(TypeError, match=f"{name} needs ax to be one Matplotlib Axes, got {type(wrong).__name__}"):
            func(data, ax=wrong, **kwargs)
    assert not any(a.has_data() for a in axes)


def test_ax_given_by_position_is_checked_too():
    with pytest.raises(TypeError, match="one Matplotlib Axes, got Figure"):
        vc.waterfall(SMALL, "k", "a", "changes", None, "Total", ("red", "green", "grey"), plt.figure())


def test_network_map_interactive_does_not_ignore_ax():
    pytest.importorskip("networkx")
    pytest.importorskip("plotly")
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="interactive=True.*cannot be drawn into ax"):
        vc.network_map(datasets.network(), "source", "target", interactive=True, ax=ax)
    assert not ax.has_data()
