"""Text for levels and columns: missing values, look-alike levels, legends, level order and error messages."""

import io
import warnings
import xml.dom.minidom

import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets
from viz_calc import stats as st
from viz_calc._core import category_order, level_label


def _ticks(ax, axis="x"):
    return [t.get_text() for t in (ax.get_xticklabels() if axis == "x" else ax.get_yticklabels())]


def _legend(ax):
    return [t.get_text().lstrip("\u200b") for t in ax.get_legend().get_texts()]


# --- missing labels ---------------------------------------------------------------------------------------------------

STEPS = pd.DataFrame({"k": pd.Series(["a", None, "b", np.nan], dtype=object), "a": [3.0, 5, 2, 8], "b": [4.0, 1, 6, 7]})


def test_level_label_reads_any_missing_value_as_missing():
    for value in (None, np.nan, pd.NA, pd.NaT, np.datetime64("NaT"), float("nan")):
        assert level_label(value) == "(missing)"
    assert level_label("nan") == "nan"  # a real level spelled like a missing value keeps its text


@pytest.mark.parametrize(("name", "draw", "axis"), [
    ("dumbbell", lambda d: vc.dumbbell(d, "k", "a", "b", sort=False).axes, "y"),
    ("waterfall", lambda d: vc.waterfall(d, label="k", value="a", total_label=None).axes, "x"),
    ("funnel", lambda d: vc.funnel(d, stage="k", value="a").axes, "y"),
    ("duration_plot", lambda d: vc.duration_plot(d.assign(s=pd.Timestamp("2024-01-01"), e=pd.Timestamp("2024-02-01")),
                                                 label="k", start="s", end="e", inner_start="s", inner_end="e",
                                                 sort=False).axes, "y"),
    ("gantt", lambda d: vc.gantt(d.assign(s=pd.Timestamp("2024-01-01"), e=pd.Timestamp("2024-02-01")),
                                 task="k", start="s", end="e", sort=False).axes, "y"),
])
def test_missing_labels_read_missing_in_one_row_per_label_charts(name, draw, axis):
    ticks = _ticks(draw(STEPS), axis)
    assert sorted(ticks) == ["(missing)", "(missing)", "a", "b"]


def test_missing_labels_in_bullet_and_circular_bar_read_missing():
    res = vc.bullet(STEPS, label="k", value="a")
    assert [_ticks(ax, "y")[0] for ax in res.axes] == ["a", "(missing)", "b", "(missing)"]
    res = vc.circular_bar(STEPS, label="k", value="a", sort=False)
    assert [t.get_text() for t in res.axes.texts] == ["a", "(missing)", "b", "(missing)"]


def test_waterfall_table_keeps_a_missing_label_missing():
    labels = vc.waterfall(STEPS, label="k", value="a").table["label"]
    assert labels.iloc[[0, 2, 4]].tolist() == ["a", "b", "Total"]
    assert labels.iloc[[1, 3]].isna().all()


def test_quadrant_plot_leaves_points_without_a_label_unannotated():
    d = datasets.measurements().head(12).assign(name=["p", None] * 6)
    res = vc.quadrant_plot(d, x="length", y="width", label="name")
    assert [t.get_text() for t in res.axes.texts if t.get_text() in ("p", "nan", "None", "(missing)")] == ["p"] * 6


# --- control characters -----------------------------------------------------------------------------------------------

def test_control_characters_in_levels_and_column_names_are_replaced():
    assert level_label("a\x00b\x1fc\x7fd\te\nf") == "a\ufffdb\ufffdc\ufffdd\te\nf"
    data = pd.DataFrame({"grp\x01": ["x\x02", "y\x1b", "x\x02", "y\x1b"] * 5, "val\x03": np.arange(20.0)})
    charts = [
        vc.lollipop(data, x="grp\x01", y="val\x03"),
        vc.stacked_percentages(data.assign(c=["p", "q"] * 10), group="grp\x01", category="c"),
        vc.histogram(data, x="val\x03", hue="grp\x01"),
        vc.raincloud(data, x="grp\x01", y="val\x03"),
        vc.waterfall(data.head(4), label="grp\x01", value="val\x03"),
    ]
    for res in charts:
        buffer = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a missing glyph would warn
            res.figure.savefig(buffer, format="svg")
        xml.dom.minidom.parseString(buffer.getvalue())  # well-formed XML
    assert sorted(_ticks(charts[0].axes, "y")) == ["x\ufffd", "y\ufffd"]
    assert charts[0].axes.get_xlabel() == "mean of val\ufffd"


def test_a_callers_own_label_text_is_kept():
    res = vc.dumbbell(STEPS.dropna(), "k", "a", "b", start_label="before\x01")
    assert _legend(res.axes)[0] == "before\x01"


# --- date labels ------------------------------------------------------------------------------------------------------

DATES = pd.DataFrame({"s": pd.to_datetime(["2024-01-01", "2024-02-01"]), "t": pd.to_datetime(["2024-02-01", "2024-03-01"])})


def test_network_map_and_sankey_label_dates_as_dates():
    pytest.importorskip("networkx")
    res = vc.network_map(DATES, "s", "t")
    assert sorted(t.get_text() for t in res.axes.texts) == ["2024-01-01", "2024-02-01", "2024-03-01"]
    pytest.importorskip("plotly")
    res = vc.network_map(DATES, "s", "t", interactive=True)
    assert sorted(res.figure.data[-1].text) == ["2024-01-01", "2024-02-01", "2024-03-01"]
    assert res.figure.data[-1].hovertext[0].startswith("<b>2024-01-01</b>")
    res = vc.sankey(DATES.assign(v=1), "s", "t", "v")
    assert list(res.figure.data[0].node.label) == ["2024-01-01", "2024-02-01", "2024-03-01"]


def test_time_zone_aware_steps_and_set_names_read_as_dates():
    stamps = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]).tz_localize("UTC")
    d = pd.DataFrame({"k": stamps, "v": [3.0, 5, 2]})
    assert _ticks(vc.waterfall(d, label="k", value="v").axes) == ["2024-01-01", "2024-02-01", "2024-03-01", "Total"]
    assert _ticks(vc.funnel(d, stage="k", value="v").axes, "y") == ["2024-01-01", "2024-02-01", "2024-03-01"]
    res = vc.upset({pd.Timestamp("2024-01-01"): [1, 2], pd.Timestamp("2024-02-01"): [2, 3]})
    assert _ticks(res.axes[2], "y") == ["2024-01-01", "2024-02-01"]


# --- category_order ---------------------------------------------------------------------------------------------------

def test_order_entries_come_back_as_the_datas_own_levels():
    d = pd.DataFrame({"g": [1, 2] * 6, "y": np.arange(12.0)})
    assert vc.benchmark_bar(d, x="g", y="y", order=[True, 2]).table["g"].tolist() == [1, 2]
    table = vc.benchmark_bar(d, x="g", y="y", order=[2.0, 1.0]).table
    assert table["g"].tolist() == [2, 1] and table["g"].dtype.kind == "i"
    m = datasets.measurements()
    m = m.assign(group=m["group"].map({"alpha": 1, "beta": 2, "gamma": 3}))
    res = vc.radar(m, ["length", "width", "depth"], "group", order=[True, 2, 3])
    assert res.table["group"].tolist() == [1, 2, 3]


def test_levels_that_cannot_be_compared_are_sorted_by_type_whatever_the_row_order():
    values = pd.Series([2020, "2019", 2018, 2.5, "2017", 2020], dtype=object)
    assert category_order(values) == category_order(values[::-1]) == [2.5, 2018, 2020, "2017", "2019"]
    d = pd.DataFrame({"g": values, "y": np.arange(6.0)})
    for frame in (d, d[::-1]):
        assert vc.raincloud(frame, x="g", y="y").table["g"].tolist() == [2.5, 2018, 2020, "2017", "2019"]
    trial = datasets.trial().assign(year=lambda t: pd.Series([2020, "2019", 2018] * 40, dtype=object, index=t.index))
    res = vc.estimation_plot(trial.sample(frac=1, random_state=0), x="year", y="score", n_resamples=200)
    assert res.info["reference"] == 2018


def test_float32_levels_read_with_their_own_digits():
    d = pd.DataFrame({"b": np.array([0.1, 0.2, 0.1, 0.3] * 15, dtype=np.float32), "y": np.arange(60.0)})
    assert category_order(d["b"]) == [np.float32(0.1), np.float32(0.2), np.float32(0.3)]
    assert _ticks(vc.profile_boxes(d, categorical=["b"], numeric=["y"]).figure[0].axes[0]) == ["0.1", "0.2", "0.3"]
    assert [t.split(":")[0] for t in _legend(vc.waffle(d, category="b").axes)] == ["0.1", "0.2", "0.3"]
    titles = [ax.get_title(loc="left") for ax in vc.compare_correlations(d.assign(z=d["y"] ** 2), "b", ["y", "z"]).axes.flat]
    assert "r(0.2) \u2212 r(0.1)" in titles


# --- NumPy scalars in messages ----------------------------------------------------------------------------------------

@pytest.mark.parametrize(("draw", "text"), [
    (lambda: vc.lollipop(pd.DataFrame({"g": [1, 2, 3] * 4, "y": np.arange(12.0)}), x="g", y="y",
                         stat=np.str_("avg")), "got 'avg'"),
    (lambda: st.mean_ci([1.0, 2.0, 4.0], np.float64(95.0)), "got 95.0"),
    (lambda: vc.bullet(pd.DataFrame({"l": ["a"], "v": [1.0]}), label="l", value="v",
                       bands=np.array([2.0, np.nan, 9.0])), "got [2.0, nan, 9.0]"),
    (lambda: vc.benchmark_bar(pd.DataFrame({"g": [1, 2] * 3, "y": np.arange(6.0)}), x="g", y="y",
                              order=np.array([1, 2, 9])), "not in the data: [9]"),
    (lambda: vc.histogram(datasets.trial(), x="score", col_wrap=np.float64(2.0)), "got the float 2.0"),
    (lambda: vc.histogram(datasets.trial(), x="score", bins=np.float32(0.1)), "got 0.1"),
])
def test_messages_show_numpy_scalars_as_python_values(draw, text):
    with pytest.raises(ValueError) as error:
        draw()
    assert text in str(error.value) and "np." not in str(error.value)


# --- levels that would be labelled alike ------------------------------------------------------------------------------

def test_a_real_missing_level_next_to_missing_values_is_rejected():
    d = pd.DataFrame({"c": ["(missing)", "a", None, "a", "(missing)", "b"], "v": [1.0, 2, 3, 4, 5, 6]})
    for draw in (lambda: vc.waffle(d, category="c"), lambda: vc.nested_pie(d.assign(i="x"), outer="c", inner="i"),
                 lambda: vc.circular_bar(d.assign(k=list("pqrstu")), label="k", value="v", group="c"),
                 lambda: vc.lollipop(d, x="c", y="v"), lambda: vc.histogram(d, x="v", hue="c")):
        with pytest.raises(ValueError, match=r"level '\(missing\)' and missing values"):
            draw()
    vc.waffle(d.dropna(), category="c")  # without missing values the level is drawn


def test_levels_that_read_alike_are_rejected():
    d = pd.DataFrame({"g": pd.Series([1, "1", 2, 1, "1", 2], dtype=object), "y": np.arange(6.0)})
    with pytest.raises(ValueError, match="would both be labelled '1'"):
        vc.benchmark_bar(d, x="g", y="y")


@pytest.mark.parametrize("kwargs", [{}, {"start_label": "b"}, {"total_label": "c"}])
def test_waterfall_rejects_a_total_named_like_a_step(kwargs):
    d = pd.DataFrame({"k": ["Total", "b", "c", "d"] if not kwargs else ["a", "b", "c", "d"], "v": [3.0, 5, 2, 8]})
    with pytest.raises(ValueError, match="also the label of a step"):
        vc.waterfall(d, label="k", value="v", **kwargs)


@pytest.mark.parametrize("labels", [["x", "x"], [None, "y"], [1, "1"]])
def test_timeseries_fill_rejects_labels_that_do_not_tell_the_series_apart(labels):
    with pytest.raises(ValueError, match="two different names"):
        vc.timeseries_fill(datasets.sales().head(60), time="date", series=["online", "store"], labels=labels)


# --- legends ----------------------------------------------------------------------------------------------------------

HIDDEN = ["_hidden", "", "shown"]


def _hidden_frame(n=60):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"g": (HIDDEN * n)[:n], "c": (["p", "q"] * n)[:n], "x": rng.normal(size=n),
                         "y": rng.normal(size=n), "z": rng.normal(size=n), "w": rng.normal(size=n)})


@pytest.mark.parametrize("name", ["stacked_percentages", "histogram", "radar", "pca_plot", "animated_bubble", "gantt",
                                  "waffle", "likert"])
def test_legends_keep_levels_that_are_empty_or_start_with_an_underscore(name):
    d = _hidden_frame()
    if name == "stacked_percentages":
        ax = vc.stacked_percentages(d, group="c", category="g").axes
    elif name == "histogram":
        ax = vc.histogram(d, x="x", hue="g").axes.flat[0]
    elif name == "radar":
        ax = vc.radar(d, ["x", "y", "z"], "g").axes
    elif name == "pca_plot":
        ax = vc.pca_plot(d, ["x", "y", "z", "w"], "g").axes
    elif name == "animated_bubble":
        ax = vc.animated_bubble(d.assign(t=[1, 2] * 30, s=1.0), x="x", y="y", size="s", time="t", color="g").axes
    elif name == "gantt":
        ax = vc.gantt(d.head(3).assign(task=["t1", "t2", "t3"], s=pd.Timestamp("2024-01-01"),
                                       e=pd.Timestamp("2024-02-01")), task="task", start="s", end="e", group="g").axes
    elif name == "waffle":
        ax = vc.waffle(d, category="g").axes
    else:
        ax = vc.likert(d.assign(q=d["g"]), items=["q"], levels=HIDDEN).axes
    texts = [t.split(":")[0] for t in _legend(ax)]
    assert sorted(texts) == sorted(HIDDEN)
