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


def test_benchmark_bar_raw_points_stay_visible_under_a_dark_style(dark_style):
    from matplotlib.collections import PathCollection

    res = vc.benchmark_bar(datasets.trial(), x="arm", y="score", threshold=50)
    points = [c for c in res.axes.collections if isinstance(c, PathCollection)]
    assert points and all(_contrast(c.get_facecolors()[0], res.axes.get_facecolor()) >= 3 for c in points)


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


def test_bullet_marks_past_the_last_band_follow_the_face(dark_style):
    data = pd.DataFrame({"kpi": ["past", "inside"], "value": [250, 150], "target": [280, 170]})
    res = vc.bullet(data, label="kpi", value="value", target="target", bands=[100, 200])
    past, inside = res.axes
    face = past.get_facecolor()
    bar = past.patches[-1]
    assert _contrast(past.get_lines()[0].get_color(), face) >= 3  # the target beyond the bands
    assert to_hex(bar.get_facecolor()) == "#222222" and _contrast(bar.get_edgecolor(), face) >= 3
    assert to_hex(inside.get_lines()[0].get_color()) == "#000000"
    assert inside.patches[-1].get_edgecolor()[3] == 0
    with plt.style.context("default"):
        light = vc.bullet(data, label="kpi", value="value", target="target", bands=[100, 200]).axes[0]
    assert to_hex(light.get_lines()[0].get_color()) == "#000000" and light.patches[-1].get_edgecolor()[3] == 0


@pytest.mark.parametrize("dark", [False, True], ids=["default style", "dark style"])
def test_nested_pie_shades_of_a_white_parent_stay_apart(dark):
    data = pd.DataFrame({"outer": np.repeat([f"P{i}" for i in range(8)], 3), "inner": ["a", "b", "c"] * 8})
    with plt.rc_context():
        if dark:
            plt.style.use("dark_background")  # the eighth default colour is white
            res = vc.nested_pie(data, outer="outer", inner="inner")
        else:
            res = vc.nested_pie(data, outer="outer", inner="inner", colors=["white", *vc.OKABE_ITO[:7]])
    wedges = [to_hex(w.get_facecolor()) for w in res.axes.patches]
    parents, shades = wedges[:8], wedges[8:]
    assert "#ffffff" in parents
    for k, parent in enumerate(parents):
        assert len({parent, *shades[3 * k:3 * k + 3]}) == 4, (parent, shades[3 * k:3 * k + 3])


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


# --- labels drawn on a fill take black or white from that fill ------------------------------------------------------

LIGHT_AND_DARK = [("#F0E442", "#08306b"), ("#08306b", "#F0E442")]


@pytest.mark.parametrize("dark", [False, True], ids=["default style", "dark style"])
@pytest.mark.parametrize("colors", [None, *LIGHT_AND_DARK])
def test_donut_grid_labels_read_on_their_wedge(dark, colors):
    data = pd.DataFrame({"a": [3, 5], "b": [4, 2]}, index=["x", "y"])
    with plt.rc_context():
        if dark:
            plt.style.use("dark_background")
        res = vc.donut_grid(data, colors=None if colors is None else list(colors))
    checked = 0
    for ax in res.figure.axes:
        for t in ax.texts:
            if t.get_text().endswith("%"):
                point = ax.transData.transform(t.get_position())
                wedge = next(w for w in ax.patches if w.contains_point(point))
                assert _contrast(t.get_color(), wedge.get_facecolor()) >= 4.5, (t.get_text(), to_hex(t.get_color()))
                checked += 1
    assert checked == 4


@pytest.mark.parametrize("colors", LIGHT_AND_DARK)
def test_centered_bar_labels_read_on_their_bar(colors):
    res = vc.centered_bar(datasets.trial(), x="arm", y="score", threshold=55, colors=colors)
    ax = res.axes
    assert len(ax.texts) == len(ax.patches) == 6  # one centred label per bar, in bar order
    for t, bar in zip(ax.texts, ax.patches):
        assert _contrast(t.get_color(), bar.get_facecolor()) >= 4.5, (t.get_text(), to_hex(t.get_color()))


def _check_matrix_labels(ax):
    im = ax.images[0]
    cells = im.get_array()
    labels = [t for t in ax.texts if t.get_text()]
    assert labels
    for t in labels:
        col, row = (int(round(v)) for v in t.get_position())
        cell = im.cmap(im.norm(cells[row, col]))
        assert _contrast(t.get_color(), cell) >= 4.5, (t.get_text(), to_hex(t.get_color()), to_hex(cell))


@pytest.mark.parametrize("cmap", ["viridis", "cividis", "RdBu_r", "PuOr_r"])
def test_matrix_labels_read_on_their_cell(cmap):
    meas = datasets.measurements()
    _check_matrix_labels(vc.correlogram(meas, method="spearman", cmap=cmap).axes)
    res = vc.compare_correlations(meas, group="group", columns=["length", "width", "depth", "mass"], cmap=cmap)
    for ax in res.axes.flat:
        if ax.get_visible():
            _check_matrix_labels(ax)


@pytest.mark.parametrize("dark", [False, True], ids=["default style", "dark style"])
def test_network_labels_read_on_their_node_or_on_the_face(dark):
    pytest.importorskip("networkx")
    from matplotlib.collections import PathCollection

    with plt.rc_context():
        if dark:
            plt.style.use("dark_background")
        res = vc.network_map(datasets.network(), source="source", target="target", weight="strength",
                             size_by="betweenness")
    ax = res.axes
    nodes = next(c for c in ax.collections if isinstance(c, PathCollection))
    order = list(res.table["node"])
    on_node = 0
    for t in ax.texts:
        k = order.index(t.get_text())
        diameter = np.sqrt(nodes.get_sizes()[k]) * res.figure.dpi / 72
        if t.get_window_extent().width <= diameter:
            on_node += 1
            assert _contrast(t.get_color(), nodes.get_facecolors()[k]) >= 4.5, (t.get_text(), to_hex(t.get_color()))
        else:  # wider than its node: most of the label is on the axes face
            assert _contrast(t.get_color(), ax.get_facecolor()) >= 4.5, t.get_text()
    assert on_node >= len(order) // 2
