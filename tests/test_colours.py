"""Colours that depend on what is under them: the axes face of a dark style, and the fill under a label."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex, to_rgb

import viz_calc as vc
from viz_calc import _core, datasets


def _luminance(colour):
    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in to_rgb(colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b):
    """WCAG contrast ratio of two colours (1 to 21)."""
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.fixture
def dark_style():
    with plt.rc_context():
        plt.style.use("dark_background")
        yield


def _monthly():
    sales = datasets.sales()
    return (sales.assign(month=sales["date"].dt.to_period("M").astype(str))
            .groupby("month", as_index=False)[["online", "store"]].sum().tail(8))


FUNNEL = pd.DataFrame({"stage": ["Visited", "Signed up", "Activated", "Paid"], "users": [10000, 3200, 1900, 410]})
STEPS = pd.DataFrame({"item": ["Opening", "Sales", "Refunds", "Costs"], "amount": [1000, 650, -120, -380]})
LIKERT = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]

# Charts whose reference lines, error bars or labels sit on the axes face; "lines" checks every Line2D and
# LineCollection, "texts" every text drawn in the axes (labels drawn on a bar are checked further down).
ON_THE_FACE = {
    "estimation_plot": (lambda: vc.estimation_plot(datasets.trial(), x="arm", y="score"), "lines"),
    "benchmark_bar": (lambda: vc.benchmark_bar(datasets.trial(), x="arm", y="score", threshold=50), "lines"),
    "lollipop": (lambda: vc.lollipop(datasets.survey(), x="team", y="tenure_years", threshold=3), "lines"),
    "divergent_bar": (lambda: vc.divergent_bar(_monthly(), category="month", left="store", right="online"), "lines"),
    "centered_bar": (lambda: vc.centered_bar(datasets.trial(), x="arm", y="score", threshold=55), "lines"),
    "likert": (lambda: vc.likert(datasets.survey(), items=["Tools are adequate"], levels=LIKERT), "lines"),
    "quadrant_plot": (lambda: vc.quadrant_plot(datasets.trial(), x="baseline", y="score"), "lines"),
    "histogram": (lambda: vc.histogram(datasets.trial(), x="score", ref_line="mean"), "lines"),
    "waterfall": (lambda: vc.waterfall(STEPS, label="item", value="amount"), "lines"),
    "funnel": (lambda: vc.funnel(FUNNEL, stage="stage", value="users"), "texts"),
    "pca_plot": (lambda: vc.pca_plot(datasets.measurements(), features=["length", "width", "depth", "mass"],
                                     group="group", loadings=True), "texts"),
}


@pytest.mark.parametrize("name", ON_THE_FACE)
def test_marks_and_labels_on_the_face_stay_visible_under_a_dark_style(dark_style, name):
    make, kind = ON_THE_FACE[name]
    res = make()
    checked = 0
    for ax in res.figure.axes:
        face = ax.get_facecolor()
        colours = []
        if kind == "lines":
            colours += [(line.get_label(), line.get_color()) for line in ax.get_lines()]
            colours += [("LineCollection", c.get_colors()[0]) for c in ax.collections
                        if isinstance(c, LineCollection) and len(c.get_colors())]
        else:
            colours += [(t.get_text(), t.get_color()) for t in ax.texts if t.get_text().strip()]
        for label, colour in colours:
            assert _contrast(colour, face) >= 3, (name, label, to_hex(colour))
        checked += len(colours)
    assert checked


def test_bullet_marks_follow_the_face_without_bands_and_stay_dark_on_them(dark_style):
    data = pd.DataFrame({"kpi": ["Revenue", "Profit"], "value": [270, -5], "target": [250, 10]})
    res = vc.bullet(data, label="kpi", value="value", target="target")
    for ax in res.axes:
        face = ax.get_facecolor()
        bar = ax.patches[-1].get_facecolor()
        marks = [line.get_color() for line in ax.get_lines()]  # the target and, below zero, the zero line
        assert _contrast(bar, face) >= 3 and all(_contrast(c, face) >= 3 for c in marks)
    banded = vc.bullet(data.assign(ok=[300, 20]), label="kpi", value="value", target="target", bands=["ok"])
    for ax in banded.axes:  # on the light grey bands the measure stays dark, as Few specifies
        assert to_hex(ax.patches[-1].get_facecolor()) == "#222222"
        assert all(to_hex(line.get_color()) == "#000000" for line in ax.get_lines())
    custom = vc.bullet(data, label="kpi", value="value", bar_color="tab:blue")
    assert to_hex(custom.axes[0].patches[-1].get_facecolor()) == to_hex("tab:blue")


def test_upset_bars_and_dots_stay_visible_under_a_dark_style(dark_style):
    res = vc.upset(datasets.memberships(), max_intersections=15)
    ax_bar, ax_mat, _ = res.axes
    face = ax_bar.get_facecolor()
    assert all(_contrast(p.get_facecolor(), face) >= 3 for p in ax_bar.patches)
    dots = [c.get_facecolors()[0] for c in ax_mat.collections]
    members, others = dots[1::2], dots[0::2]  # each column draws every set's dot, then the members' on top
    assert all(_contrast(c, face) >= 3 for c in members)
    assert all(_contrast(o, face) < _contrast(m, face) for o, m in zip(others, members))
    assert all(_contrast(line.get_color(), face) >= 3 for line in ax_mat.get_lines())
    custom = vc.upset(datasets.memberships(), color="tab:red")
    assert to_hex(custom.axes[0].patches[0].get_facecolor()) == to_hex("tab:red")


def test_eighth_default_colour_is_white_under_a_dark_style(dark_style):
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"group": np.repeat(list("ABCDEFGH"), 10),
                         **{m: rng.normal(size=80) for m in ["a", "b", "c", "d"]}})
    res = vc.radar(data, metrics=["a", "b", "c", "d"], group="group")
    face = res.axes.get_facecolor()
    assert all(_contrast(line.get_color(), face) >= 3 for line in res.axes.get_lines())
    assert vc.palette(8)[7] == "#ffffff"
    with plt.style.context("default"):
        assert vc.palette(8) == list(vc.OKABE_ITO)


def test_plotly_colours_do_not_follow_the_matplotlib_style(dark_style):
    pytest.importorskip("plotly")
    pytest.importorskip("networkx")
    flows = pd.DataFrame({"s": list("ABCD"), "t": list("EFGH"), "v": [1, 2, 3, 4]})
    assert list(vc.sankey(flows, source="s", target="t", value="v").figure.data[0].node.color) == list(vc.OKABE_ITO)
    pairs = pd.DataFrame({"s": [f"a{i}" for i in range(8)], "t": [f"b{i}" for i in range(8)]})
    res = vc.network_map(pairs, source="s", target="t", interactive=True)
    assert res.info["n_communities"] == 8
    assert "#000000" in list(res.figure.data[-1].marker.color)


def test_gridlines_and_zero_lines_stay_faint_under_a_dark_style(dark_style):
    proj = datasets.projects()
    charts = {
        "dumbbell": vc.dumbbell(_monthly(), label="month", start="store", end="online", show_values=True),
        "gantt": vc.gantt(proj, task="task", start="start", end="end", group="team"),
        "duration_plot": vc.duration_plot(proj, label="task", start="start", end="end", inner_start="work_start",
                                          inner_end="work_end"),
    }
    for name, res in charts.items():
        ax = res.axes
        grid = [line.get_color() for line in ax.get_xgridlines() if line.get_visible()]
        assert grid, name
        assert all(_contrast(c, ax.get_facecolor()) < 1.5 for c in grid), (name, grid)
    pca = vc.pca_plot(datasets.measurements(), features=["length", "width", "depth", "mass"])
    zero_lines = [line.get_color() for line in pca.axes.get_lines()]
    assert zero_lines and all(_contrast(c, pca.axes.get_facecolor()) < 1.5 for c in zero_lines)


def test_quadrant_label_boxes_match_a_dark_face(dark_style):
    res = vc.quadrant_plot(datasets.trial(), x="baseline", y="score")
    boxes = [t.get_bbox_patch().get_facecolor() for t in res.axes.texts]
    assert len(boxes) == 4 and all(_luminance(b) < 0.05 for b in boxes)


def test_the_default_style_keeps_its_colours():
    with plt.style.context("default"):
        est = vc.estimation_plot(datasets.trial(), x="arm", y="score")
        assert {to_hex(c.get_colors()[0]) for ax in est.axes for c in ax.collections
                if isinstance(c, LineCollection)} == {"#000000"}
        up = vc.upset(datasets.memberships())
        assert to_hex(up.axes[0].patches[0].get_facecolor()) == "#333333"
        assert {to_hex(c.get_facecolors()[0]) for c in up.axes[1].collections} == {"#333333", "#dddddd"}
        bul = vc.bullet(pd.DataFrame({"k": ["a"], "v": [3], "t": [4]}), label="k", value="v", target="t")
        assert to_hex(bul.axes[0].patches[-1].get_facecolor()) == "#222222"
        gantt = vc.gantt(datasets.projects(), task="task", start="start", end="end")
        assert {to_hex(line.get_color()) for line in gantt.axes.get_xgridlines()} == {"#eeeeee"}


def test_foreground_follows_the_face_it_is_drawn_on():
    fig, ax = plt.subplots()
    assert _core.foreground(ax) == "black" and _core.foreground(ax, "#eeeeee") == "#eeeeee"
    ax.set_facecolor("#e5e5e5")  # ggplot's grey face is still a light face
    assert _core.foreground(ax) == "black"
    ax.set_facecolor("black")
    assert [_core.foreground(ax, c) for c in ("black", "#eeeeee", "#333333")] == ["#ffffff", "#111111", "#cccccc"]
    ax.set_facecolor("none")  # a transparent axes shows the figure behind it
    fig.set_facecolor("#202020")
    assert _core.foreground(ax) == "#ffffff"
    fig.set_facecolor("none")
    assert _core.foreground(ax) == "black"
    with plt.rc_context({"text.color": "white"}):  # nothing opaque behind: light text means a dark page
        assert _core.foreground(ax) == "#ffffff"
