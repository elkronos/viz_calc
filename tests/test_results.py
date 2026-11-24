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
def test_a_3d_ax_raises_the_projection_error_before_drawing(name):
    func, data, kwargs = _ax_call(name)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")  # Matplotlib 3.10 and later give it no get_frame_on
    before = len(ax.get_children()), len(fig.get_children())
    with pytest.raises(ValueError, match="needs a polar Axes" if AX_CALLS[name][3] else r"\(Cartesian\) Axes, got a '3d'"):
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


# --- to_pptx --------------------------------------------------------------------------------------------------------


def _slide_texts(deck):
    pptx = pytest.importorskip("pptx")
    return [[sh.text_frame for sh in s.shapes if sh.has_text_frame] for s in pptx.Presentation(deck).slides]


def test_to_pptx_returns_a_stream_target_as_given_and_a_path_as_text(tmp_path):
    pytest.importorskip("pptx")
    res = vc.waffle(datasets.survey(), category="team")
    buf = BytesIO()
    assert vc.to_pptx(res, buf) is buf and buf.getvalue().startswith(b"PK")
    assert vc.to_pptx(res, tmp_path / "deck.pptx") == str(tmp_path / "deck.pptx")


def test_to_pptx_shows_any_title_as_text_and_sets_every_line_in_24_pt():
    pytest.importorskip("pptx")
    res = vc.waffle(datasets.survey(), category="team")
    deck = vc.to_pptx([res, res, res, res, res], BytesIO(), titles=[2024, 0, 3.5, None, "Line one\nLine two"])
    frames = _slide_texts(deck)
    assert [[f.text for f in slide] for slide in frames] == [["2024"], ["0"], ["3.5"], [], ["Line one\nLine two"]]
    sizes = [run.font.size.pt for p in frames[4][0].paragraphs for run in p.runs]
    assert sizes == [24, 24]


def test_to_pptx_rejects_more_titles_than_items():
    pytest.importorskip("pptx")
    pages = vc.profile_boxes(datasets.measurements(), per_page=2)
    with pytest.raises(ValueError, match="got 3 titles for 1 item.*multi-page result takes one title"):
        vc.to_pptx(pages, BytesIO(), titles=["A", "B", "C"])


def test_to_pptx_explains_a_dpi_too_low_for_the_text():
    pytest.importorskip("pptx")
    with pytest.raises(ValueError, match="dpi=5 is too low to draw the text of this figure"):
        vc.to_pptx(vc.profile_bars(datasets.measurements()), BytesIO(), dpi=5)


def test_to_pptx_checks_the_image_size_itself_and_hides_pillows_warning(monkeypatch):
    import struct
    import warnings

    pytest.importorskip("pptx")
    from PIL import Image

    res = vc.histogram(datasets.measurements(), "length")
    png = BytesIO()
    res.figure.savefig(png, format="png", dpi=100, bbox_inches="tight")
    width, height = struct.unpack(">II", png.getvalue()[16:24])
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", width * height // 3)  # Pillow refuses more than twice this
    with pytest.raises(ValueError, match=f"dpi=100 makes a {width} x {height} pixel image, more than the"):
        vc.to_pptx(res, BytesIO(), dpi=100)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", width * height * 2 // 3)  # Pillow only warns
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        assert len(_slide_texts(vc.to_pptx(res, BytesIO(), dpi=100))) == 1


# --- AbbrevFormatter outside an Axes --------------------------------------------------------------------------------


def test_abbrev_formatter_works_on_a_dummy_axis():
    from viz_calc._core import AbbrevFormatter

    fmt = AbbrevFormatter()
    fmt.create_dummy_axis()
    assert fmt.format_ticks([10000, 20000, 30000]) == ["10K", "20K", "30K"]
    points = plt.figure().add_subplot().scatter([1, 2, 3], [1, 2, 3], c=[10000, 20000, 30000])
    _, labels = points.legend_elements(fmt=AbbrevFormatter())
    assert labels and all("K" in label for label in labels)


# --- the user's layout settings -------------------------------------------------------------------------------------


def test_compare_correlations_draws_under_a_constrained_layout_setting():
    import warnings

    meas = datasets.measurements()
    with plt.rc_context({"figure.constrained_layout.use": True}), warnings.catch_warnings():
        warnings.simplefilter("error")
        res = vc.compare_correlations(meas, group="group")
        res.save(BytesIO(), format="png")
    assert type(res.figure.get_layout_engine()).__name__ == "ConstrainedLayoutEngine"
    width = res.figure.get_window_extent().width
    boxes = [ax.get_window_extent() for ax in res.figure.axes]
    assert all(0 <= b.x0 and b.x1 <= width for b in boxes)  # every heatmap and colour bar inside the figure
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])


def _own_layout_calls():
    """``name: (function, data, keyword arguments)`` for the charts that lay out their own figure with tight_layout."""
    trial, meas, sales = datasets.trial(), datasets.measurements(), datasets.sales()
    bubbles = sales.head(20).melt(id_vars="date", value_vars=["online", "store"]).assign(size=lambda d: d["value"])
    return {
        "estimation_plot": (vc.estimation_plot, trial, dict(x="arm", y="score", n_resamples=200)),
        "histogram": (vc.histogram, trial, dict(x="score", facet="arm")),
        "percent_grid": (vc.percent_grid, trial, dict(column="improved")),
        "donut_grid": (vc.donut_grid, SMALL.set_index("k"), dict()),
        "bullet": (vc.bullet, SMALL, dict(label="k", value="a", target="b")),
        "animated_bubble": (vc.animated_bubble, bubbles, dict(time="date", x="value", y="size", size="size",
                                                              color="variable")),
        "profile_bars": (vc.profile_bars, meas, dict()),
        "profile_boxes": (vc.profile_boxes, meas, dict()),
        "profile_scatters": (vc.profile_scatters, meas, dict()),
    }


OWN_LAYOUT = _own_layout_calls()


@pytest.mark.parametrize("name", list(OWN_LAYOUT))
def test_a_chart_keeps_its_layout_without_a_warning_under_a_constrained_layout_setting(name):
    import warnings

    func, data, kwargs = OWN_LAYOUT[name]

    def positions():
        res = func(data, **kwargs)
        return [ax.get_position().bounds for fig in (res.figure if isinstance(res.figure, list) else [res.figure])
                for ax in fig.axes]

    usual = positions()
    with plt.rc_context({"figure.constrained_layout.use": True}), warnings.catch_warnings():
        warnings.simplefilter("error")  # Matplotlib's "The figure layout has changed to tight"
        assert positions() == usual


# --- the VizResult docstring matches the results --------------------------------------------------------------------


def test_vizresult_docstring_describes_figure_axes_and_info_as_returned():
    import re

    import numpy as np

    doc = " ".join(vc.VizResult.__doc__.split())
    one_d = re.search(r"a 1-D array for (.*?), a 2-D grid for (.*?)\. ", doc)
    assert one_d, doc
    runs = {
        "estimation_plot": lambda: vc.estimation_plot(datasets.trial(), x="arm", y="score", n_resamples=200),
        "upset": lambda: vc.upset(pd.DataFrame({"A": [1, 1, 0], "B": [0, 1, 1]}).astype(bool)),
        "bullet": lambda: vc.bullet(SMALL.head(1), label="k", value="a", ax=plt.figure().subplots()),
        "calendar_heatmap": lambda: vc.calendar_heatmap(datasets.sales().head(30), date="date", value="online"),
        "histogram": lambda: vc.histogram(datasets.trial(), x="score"),
        "percent_grid": lambda: vc.percent_grid(datasets.trial(), column="improved"),
        "donut_grid": lambda: vc.donut_grid(SMALL.head(1).set_index("k")),
        "compare_correlations": lambda: vc.compare_correlations(datasets.measurements(), group="group"),
    }
    for listed, ndim in ((one_d.group(1), 1), (one_d.group(2), 2)):
        names = re.findall(r"``(\w+)``", listed)
        assert names and set(names) <= set(runs)
        for name in names:
            axes = runs[name]().axes
            assert isinstance(axes, np.ndarray) and axes.ndim == ndim, name
    pages = vc.profile_boxes(datasets.measurements())
    assert isinstance(pages.figure, list) and pages.axes is None
    assert "a list of them for the multi-page results" in doc and "``None`` for Plotly output and for multi-page" in doc
    info = vc.pca_plot(datasets.measurements(), features=METRICS).info
    assert isinstance(info["scores"], pd.DataFrame) and isinstance(info["eigenvalues"], pd.Series)
    assert "tables (``DataFrame`` and ``Series``)" in doc and "scalar" not in doc


# --- warnings hidden while dates are parsed -------------------------------------------------------------------------


def test_date_parsing_hides_only_pandas_mixed_offset_warning(monkeypatch):
    import warnings

    to_datetime = pd.to_datetime

    def warns(values, **kwargs):
        warnings.warn("In a future version of pandas, parsing datetimes with mixed time zones will raise",
                      FutureWarning, stacklevel=2)
        warnings.warn("an unrelated deprecation", FutureWarning, stacklevel=2)
        return to_datetime(values, **kwargs)

    data = pd.DataFrame({"d": ["2021-01-01", "2021-01-15", "2021-02-01"], "v": [1.0, 2, 3]})
    monkeypatch.setattr(pd, "to_datetime", warns)
    with pytest.warns(FutureWarning, match="an unrelated deprecation") as seen:
        vc.period_bars(data, date="d", value="v")
    assert not [w for w in seen if "mixed time zones" in str(w.message)]


def test_reading_frame_labels_hides_only_dateutils_time_zone_warning(monkeypatch):
    import warnings

    from dateutil import parser

    from viz_calc import timeseries

    parse = parser.parse

    def warns(text, **kwargs):
        warnings.warn("tzname EST identified but not understood", parser.UnknownTimezoneWarning, stacklevel=2)
        warnings.warn("an unrelated warning", UserWarning, stacklevel=2)
        return parse(text, **kwargs)

    monkeypatch.setattr(parser, "parse", warns)
    with pytest.warns(UserWarning, match="an unrelated warning") as seen:
        assert timeseries._has_year("2021-03-04") is True
    assert not [w for w in seen if issubclass(w.category, parser.UnknownTimezoneWarning)]
