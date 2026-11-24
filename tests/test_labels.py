"""Text for levels and columns: missing values, look-alike levels, legends, level order and error messages."""

import io
import warnings
import xml.dom.minidom

import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets
from viz_calc._core import level_label


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
