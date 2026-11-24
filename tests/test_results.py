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


# --- which figures a failed call closes ------------------------------------------------------------------------------


def test_a_failed_call_leaves_open_the_figures_another_thread_makes_meanwhile():
    import threading

    from viz_calc._core import cleanup_on_error

    elsewhere, call_started, other_done = [], threading.Event(), threading.Event()

    def other_thread():
        call_started.wait(10)
        elsewhere.append(plt.figure())
        other_done.set()

    @cleanup_on_error
    def fails():
        plt.subplots()
        call_started.set()
        other_done.wait(10)
        raise RuntimeError("boom")

    thread = threading.Thread(target=other_thread)
    thread.start()
    with pytest.raises(RuntimeError, match="boom"):
        fails()
    thread.join()
    assert plt.get_fignums() == [elsewhere[0].number]
    assert plt.rcParams["figure.hooks"] == []


def test_a_failed_call_closes_the_figures_of_the_plotting_calls_it_made():
    from viz_calc._core import cleanup_on_error

    @cleanup_on_error
    def inner(fail):
        plt.figure()
        if fail:
            raise ValueError("inner")

    @cleanup_on_error
    def outer():
        own = plt.figure()
        with pytest.raises(ValueError, match="inner"):
            inner(fail=True)
        assert plt.get_fignums() == [keep.number, own.number]  # the inner call closed only its own figure
        inner(fail=False)
        raise RuntimeError("outer")

    keep = plt.figure()
    with pytest.raises(RuntimeError, match="outer"):
        outer()
    assert plt.get_fignums() == [keep.number]


# --- a failed call takes back what it drew on the caller's Axes ------------------------------------------------------


def _pixels(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=40)
    buf.seek(0)
    return plt.imread(buf)


def _fail_at_the_end(monkeypatch, func):
    """Make *func* fail at its last step, after it has drawn everything."""
    import sys

    def fails(*args, **kwargs):
        raise RuntimeError("failed after drawing")

    monkeypatch.setattr(sys.modules[func.__wrapped__.__module__], "VizResult", fails)


@pytest.mark.parametrize("content", [False, True], ids=["empty", "with content"])
@pytest.mark.parametrize("name", list(AX_CALLS))
def test_a_call_that_fails_after_drawing_leaves_the_callers_axes_as_they_were(name, content, monkeypatch):
    import numpy as np

    func, data, kwargs = _ax_call(name)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 2, 1, projection="polar" if AX_CALLS[name][3] else None)
    fig.add_subplot(1, 2, 2).plot([1, 2], [3, 1])
    if content:
        ax.plot([0, 1, 2], [1, 3, 2], label="mine")
        ax.set_title("My title")
        ax.set_xlabel("x")
        ax.legend()
    before, children, axes = _pixels(fig), ax.get_children(), fig.axes
    _fail_at_the_end(monkeypatch, func)
    with pytest.raises(RuntimeError, match="failed after drawing"):
        func(data, ax=ax, **kwargs)
    assert ax.get_children() == children and fig.axes == axes
    assert np.array_equal(_pixels(fig), before)


@pytest.mark.parametrize("where", ["add_axes", "subfigure", "styled"])
def test_a_failed_call_removes_its_colour_bar_and_gives_the_room_back(where, monkeypatch):
    import numpy as np

    fig = plt.figure(figsize=(6, 4))
    if where == "add_axes":
        ax = fig.add_axes([0.1, 0.1, 0.5, 0.8])
    elif where == "subfigure":
        ax = fig.subfigures(1, 2)[0].subplots()
    else:
        ax = fig.subplots()
        ax.scatter([1, 2, 30], [4, 5, 60], label="points")
        ax.set_xticks([1, 10, 30], ["one", "ten", "thirty"], rotation=30, color="red")
        ax.grid(True)
        ax.tick_params(labelsize=6, length=8)
        ax.legend()
    before, position = _pixels(fig), ax.get_position().bounds
    _fail_at_the_end(monkeypatch, vc.correlogram)
    with pytest.raises(RuntimeError):
        vc.correlogram(datasets.measurements(), ax=ax)
    assert len(fig.axes) == 1 and ax.get_position().bounds == position
    assert np.array_equal(_pixels(fig), before)


# --- VizResult.save paths -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("pages", [False, True], ids=["one figure", "pages"])
def test_save_rejects_a_path_that_does_not_name_a_file(pages, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    meas = datasets.measurements()
    res = vc.profile_boxes(meas, per_page=2) if pages else vc.histogram(meas, "length")
    (tmp_path / "pages").mkdir()
    with pytest.raises(ValueError, match="path is empty"):
        res.save("")
    for path in ("pages", "pages/", tmp_path / "pages", str(tmp_path / "missing") + "/"):
        with pytest.raises(ValueError, match="is a directory, not a file; give a file in it, such as .*chart.png"):
            res.save(path)
    assert sorted(p.name for p in tmp_path.rglob("*")) == ["pages"]  # nothing was written


def test_save_of_pages_returns_one_path_per_page_of_this_result(tmp_path):
    meas = datasets.measurements()
    four = vc.profile_boxes(meas, per_page=1)
    assert len(four.figure) > 2
    four.save(tmp_path / "boxes.png")
    two = vc.VizResult(four.figure[:2])
    assert two.save(tmp_path / "boxes.png") == [str(tmp_path / "boxes_1.png"), str(tmp_path / "boxes_2.png")]
    assert (tmp_path / "boxes_3.png").exists()  # left from the longer save, as the docstring says
