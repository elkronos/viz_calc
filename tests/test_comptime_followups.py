"""Part-to-whole and timeline charts: missing categories, shared columns, overflow and undefined rates."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import viz_calc as vc
from viz_calc import datasets
from viz_calc._core import NEUTRAL

SALES = pd.DataFrame({"region": ["N", "S", None, None], "product": ["a", "b", "a", None], "v": [10, 10, 40, 40]})


# --- one column passed for two roles --------------------------------------------------------------------------------


def test_gantt_rejects_one_column_for_start_and_end():
    with pytest.raises(ValueError, match="start and end must be different columns"):
        vc.gantt(datasets.projects(), task="task", start="start", end="start")


def test_gantt_rejects_a_date_column_as_task_or_group():
    proj = datasets.projects()
    with pytest.raises(ValueError, match="task and start must be different columns"):
        vc.gantt(proj, task="start", start="start", end="end")
    with pytest.raises(ValueError, match="group and end must be different columns"):
        vc.gantt(proj, task="task", start="start", end="end", group="end")


def test_gantt_may_colour_each_task_by_its_own_name():
    res = vc.gantt(datasets.projects(), task="task", start="start", end="end", group="task")
    assert list(res.table.columns) == ["task", "start", "end", "duration_days"]


@pytest.mark.parametrize("roles, message", [
    (("task", "start", "start", "start", "end"), "start and end"),
    (("end", "start", "end", "start", "end"), "label and end"),
    (("task", "start", "end", "end", "end"), "inner_start and inner_end"),
    (("task", "start", "end", "end", "x"), "end and inner_start"),
    (("task", "start", "end", "x", "start"), "start and inner_end"),
])
def test_duration_plot_rejects_one_column_for_two_roles(roles, message):
    proj = datasets.projects().assign(x=lambda d: d["end"])
    with pytest.raises(ValueError, match=f"{message} must be different columns"):
        vc.duration_plot(proj, *roles)


def test_duration_plot_inner_window_may_reuse_the_whole_period():
    proj = datasets.projects()
    res = vc.duration_plot(proj, label="task", start="start", end="end", inner_start="start", inner_end="end")
    assert list(res.table.columns) == ["task", "start", "end", "outer_days", "inner_days", "inner_share"]
    assert (res.table["inner_days"] == res.table["outer_days"]).all()
    assert (res.table["inner_share"].dropna() == 1).all()


# --- missing categories are kept -------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [object, "category", "string"])
def test_nested_pie_keeps_rows_with_a_missing_category(dtype):
    data = SALES.astype({"region": dtype, "product": dtype})
    res = vc.nested_pie(data, "region", "product", "v")
    t = res.table
    assert t["value"].sum() == 100
    assert t["percent_of_total"].tolist() == [10.0, 10.0, 40.0, 40.0]
    assert t["region"].isna().tolist() == [False, False, True, True]  # the missing parent comes last
    assert t["percent_of_parent"].tolist() == [100.0, 100.0, 50.0, 50.0]
    texts = [x.get_text() for x in res.axes.texts]
    assert texts == ["N", "S", "(missing)", "a", "b", "a", "(missing)"]
    inner_ring = res.axes.patches[3:]  # three parent wedges, then the children
    assert to_hex(inner_ring[-1].get_facecolor()) == to_hex(NEUTRAL)
    assert to_hex(res.axes.patches[2].get_facecolor()) == to_hex(NEUTRAL)


def test_nested_pie_missing_inner_under_a_known_parent():
    data = pd.DataFrame({"o": ["x", "x", "y"], "i": ["p", None, "q"]})
    t = vc.nested_pie(data, "o", "i").table
    assert t["value"].tolist() == [1, 1, 1]
    assert t["percent_of_parent"].tolist() == [50.0, 50.0, 100.0]


@pytest.mark.parametrize("dtype", [object, "category", "string"])
def test_waffle_keeps_rows_with_a_missing_category(dtype):
    data = SALES.astype({"region": dtype})
    res = vc.waffle(data, "region", "v")
    t = res.table
    assert t["value"].tolist() == [10, 10, 80]
    assert t["percent"].tolist() == [10.0, 10.0, 80.0]
    assert t["tiles"].tolist() == [10, 10, 80]
    assert t["region"].isna().tolist() == [False, False, True]
    legend = [x.get_text() for x in res.axes.get_legend().get_texts()]
    assert legend[-1] == "(missing): 80 (80.0%)"
    assert to_hex(res.axes.patches[-1].get_facecolor()) == to_hex(NEUTRAL)


def test_waffle_counts_and_order_include_missing_category():
    t = vc.waffle(SALES, "region").table
    assert t["value"].tolist() == [1, 1, 2]
    t = vc.waffle(SALES, "region", "v", order=["S", "N"]).table
    assert t["region"].tolist()[:2] == ["S", "N"] and t["percent"].tolist() == [10.0, 10.0, 80.0]


def test_waffle_positive_total_only_in_rows_with_missing_category():
    data = pd.DataFrame({"c": ["a", None], "v": [0, 5]})
    assert vc.waffle(data, "c", "v").table["tiles"].tolist() == [0, 100]


# --- bullet ----------------------------------------------------------------------------------------------------------


def test_bullet_integer_band_labels_are_columns():
    data = pd.DataFrame({0: ["a", "b"], 1: [3.0, 5.0], 2: [4.0, 6.0], 3: [8.0, 9.0]})
    res = vc.bullet(data, label=0, value=1, bands=[2, 3])
    bands = [(float(p.get_x()), float(p.get_x() + p.get_width())) for p in res.axes[0].patches[:2]]
    assert bands == [(0.0, 4.0), (4.0, 8.0)]


def test_bullet_numeric_limits_that_are_not_all_columns_stay_limits():
    data = pd.DataFrame({0: ["a"], 1: [3.0], 2: [4.0]})
    res = vc.bullet(data, label=0, value=1, bands=[2, 5])
    bands = [(float(p.get_x()), float(p.get_x() + p.get_width())) for p in res.axes[0].patches[:2]]
    assert bands == [(0.0, 2.0), (2.0, 5.0)]


@pytest.mark.parametrize("name", ["value", "target", "pct_of_target"])
def test_bullet_rejects_label_named_like_an_output_column(name):
    data = pd.DataFrame({name: ["Revenue", "Profit"], "actual": [270, 22], "goal": [250, 26]})
    with pytest.raises(ValueError, match="rename the label column"):
        vc.bullet(data, label=name, value="actual", target="goal")


def test_bullet_label_named_target_without_target_is_kept():
    data = pd.DataFrame({"target": ["Revenue", "Profit"], "actual": [270, 22]})
    t = vc.bullet(data, label="target", value="actual").table
    assert t["target"].tolist() == ["Revenue", "Profit"]


# --- funnel and period_bars ------------------------------------------------------------------------------------------


def test_funnel_undefined_rates_are_not_printed_as_nan():
    res = vc.funnel(pd.DataFrame({"stage": ["visit", "signup", "buy"], "n": [0, 0, 0]}), "stage", "n")
    texts = [x.get_text() for x in res.axes.texts]
    assert not any("nan" in x for x in texts)
    assert texts[:3] == ["0  (–)", "0  (–)", "↓ – of previous"]
    assert res.table["pct_of_first"].isna().all()


def test_funnel_rate_after_a_zero_stage():
    res = vc.funnel(pd.DataFrame({"stage": list("abc"), "n": [10, 0, 0]}), "stage", "n")
    texts = [x.get_text() for x in res.axes.texts]
    assert "↓ 0% of previous" in texts and "↓ – of previous" in texts


def test_period_bars_int64_sum_does_not_wrap():
    data = pd.DataFrame({"d": ["2024-01-01", "2024-01-02", "2024-02-01"],
                         "v": np.array([5 * 10**18, 5 * 10**18, 3], dtype=np.int64)})
    res = vc.period_bars(data, "d", "v")
    assert res.table["value"].tolist() == [1e19, 3.0]
    assert res.axes.patches[0].get_height() == 1e19


def test_period_bars_small_int_sums_stay_exact_integers():
    data = pd.DataFrame({"d": ["2024-01-01", "2024-01-02"], "v": np.array([2**53, 1], dtype=np.int64)})
    assert vc.period_bars(data, "d", "v").table["value"].tolist() == [2**53 + 1]
