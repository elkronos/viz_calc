"""Part-to-whole and time charts: reserved and repeated columns, date labels, boolean labels, options and big sums."""

import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets

BIG = np.array([2**62, 2**62, 5], dtype=np.int64)  # the first two sum to 2**63, past the int64 range
DAYS = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])


def _bubbles(t):
    n = len(t)
    return pd.DataFrame({"t": t, "x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float), "s": np.ones(n)})


# --- columns named like the table's own columns ---------------------------------------------------------------------


@pytest.mark.parametrize("role,name", [("task", "duration_days"), ("group", "duration_days")])
def test_gantt_rejects_a_column_named_like_its_output(role, name):
    proj = datasets.projects().rename(columns={"task" if role == "task" else "team": name})
    kwargs = {"task": name if role == "task" else "task", "group": name if role == "group" else None}
    with pytest.raises(ValueError, match=f"rename the {role} column 'duration_days'"):
        vc.gantt(proj, start="start", end="end", **kwargs)


@pytest.mark.parametrize("name", ["outer_days", "inner_days", "inner_share"])
def test_duration_plot_rejects_a_label_named_like_its_output(name):
    proj = datasets.projects().rename(columns={"task": name})
    with pytest.raises(ValueError, match=f"rename the label column '{name}'"):
        vc.duration_plot(proj, name, "start", "end", "work_start", "work_end")


def test_duration_plot_rejects_a_date_column_named_like_its_output():
    proj = datasets.projects().rename(columns={"work_end": "inner_days"})
    with pytest.raises(ValueError, match="rename the inner_end column 'inner_days'"):
        vc.duration_plot(proj, "task", "start", "end", "work_start", "inner_days")


@pytest.mark.parametrize("columns,time,series", [
    (["t", "difference", "b"], "t", ["difference", "b"]),
    (["t", "a", "leader"], "t", ["a", "leader"]),
    (["leader", "a", "b"], "leader", ["a", "b"]),
])
def test_timeseries_fill_rejects_a_column_named_like_its_output(columns, time, series):
    d = pd.DataFrame(dict(zip(columns, [[1, 2, 3, 4], [1, 3, 2, 5.0], [2, 2, 2, 2.0]])))
    with pytest.raises(ValueError, match="the table uses \\['difference', 'leader'\\]"):
        vc.timeseries_fill(d, time=time, series=series)


def test_timeseries_fill_leader_keeps_non_text_series_names():
    d = pd.DataFrame({"t": [1, 2, 3], 0: [1, 3.0, np.nan], "b": [2, 2, 2.0]})
    assert vc.timeseries_fill(d, "t", [0, "b"]).table["leader"].tolist() == ["b", 0, None]


# --- repeated columns -----------------------------------------------------------------------------------------------


def test_donut_grid_rejects_a_repeated_column():
    d = pd.DataFrame({"a": [1, 3], "b": [1, 1]})
    with pytest.raises(ValueError, match=r"columns repeats column\(s\): \['a'\]"):
        vc.donut_grid(d, columns=["a", "a", "b"])


# --- text dates in animated_bubble ----------------------------------------------------------------------------------


@pytest.mark.parametrize("labels", [
    ["05/01/2021", "12/01/2021", "19/01/2021", "26/01/2021", "02/02/2021"],
    ["13/01/2021", "20/01/2021", "27/01/2021", "03/02/2021", "10/02/2021"],
    ["05.01.2021", "19.01.2021", "02.02.2021"],
])
def test_animated_bubble_plays_day_first_dates_in_date_order(labels):
    shuffled = labels[::-1]
    res = vc.animated_bubble(_bubbles(shuffled), "t", "x", "y", "s")
    assert res.info["frames"] == labels
    assert res.table["t"].tolist() == labels


def test_animated_bubble_keeps_month_first_dates():
    labels = ["9/1/2020", "10/1/2020", "11/1/2020", "12/1/2020", "1/1/2021"]
    assert vc.animated_bubble(_bubbles(labels[::-1]), "t", "x", "y", "s").info["frames"] == labels


def test_animated_bubble_rejects_dates_that_read_both_ways_in_different_orders():
    with pytest.raises(ValueError, match="read both month first and day first"):
        vc.animated_bubble(_bubbles(["05/01/2021", "02/02/2021", "12/01/2021"]), "t", "x", "y", "s")


def test_animated_bubble_rejects_a_mixture_of_date_formats():
    with pytest.raises(ValueError, match="reads as dates but not in one consistent format"):
        vc.animated_bubble(_bubbles(["01/13/2021", "13/01/2021", "02/01/2021"]), "t", "x", "y", "s")


@pytest.mark.parametrize("labels,frames", [
    (["1/2", "1/10", "1/20", "2/1"], ["1/10", "1/2", "1/20", "2/1"]),
    (["9am", "10am", "1pm", "11am"], ["10am", "11am", "1pm", "9am"]),
    (["Phase B", "Jan 2020", "Phase A"], ["Jan 2020", "Phase A", "Phase B"]),
])
def test_animated_bubble_sorts_labels_without_a_year_as_text(labels, frames):
    assert vc.animated_bubble(_bubbles(labels), "t", "x", "y", "s").info["frames"] == frames


# --- columns labelled True / False ----------------------------------------------------------------------------------


def test_waffle_accepts_a_value_column_labelled_true():
    wide = pd.DataFrame({"team": ["a", "b", None], False: [1, 2, 3], True: [3, 1, 4]})
    res = vc.waffle(wide, "team", value=True)
    assert res.table["value"].tolist() == [3, 1, 4]


def test_funnel_and_circular_bar_accept_columns_labelled_true_and_false():
    d = pd.DataFrame({True: ["x", "y", "z"], False: [3, 2, 1]})
    assert vc.funnel(d, True, False).table["pct_of_previous"].tolist()[1:] == pytest.approx([200 / 3, 50.0])
    assert vc.circular_bar(d, True, False).table[True].tolist() == ["x", "y", "z"]


def test_animated_bubble_accepts_x_and_y_labelled_true_and_false():
    d = pd.DataFrame({"t": ["2020", "2021", "2021"], True: [1.0, 2, 3], False: [4.0, 5, 6]})
    res = vc.animated_bubble(d, "t", True, False, True)
    assert res.table.to_dict("list") == {"t": ["2020", "2021"], True: [1.0, 2.5], False: [4.0, 5.5]}


# --- numeric options ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [-0.1, 1.5, 5, np.nan, "0.2", True])
def test_timeseries_fill_rejects_a_bad_alpha(alpha):
    d = pd.DataFrame({"t": [1, 2], "a": [1, 3.0], "b": [2, 2.0]})
    with pytest.raises(ValueError, match="alpha must be a number from 0 to 1"):
        vc.timeseries_fill(d, "t", ["a", "b"], alpha=alpha)


@pytest.mark.parametrize("min_label", ["x", None, np.nan, np.inf])
def test_min_label_must_be_a_finite_number(min_label):
    d = pd.DataFrame({"a": [1, 3], "b": [1, 1]})
    with pytest.raises(ValueError, match="min_label must be a finite number"):
        vc.donut_grid(d, min_label=min_label)
    with pytest.raises(ValueError, match="min_label must be a finite number"):
        vc.stacked_percentages(d, "a", "b", min_label=min_label)


@pytest.mark.parametrize("inner_radius", [-0.1, np.nan, "a", True])
def test_circular_bar_rejects_a_bad_inner_radius(inner_radius):
    d = pd.DataFrame({"l": ["x", "y"], "v": [1, 2]})
    with pytest.raises(ValueError, match="inner_radius must be a finite number >= 0"):
        vc.circular_bar(d, "l", "v", inner_radius=inner_radius)


# --- integer sums past the int64 range ------------------------------------------------------------------------------


def test_calendar_heatmap_sums_large_integers_without_wrapping():
    res = vc.calendar_heatmap(pd.DataFrame({"d": DAYS, "v": BIG}), "d", "v")
    assert res.table["value"].tolist() == [2.0**63, 5.0]


def test_nested_pie_sums_large_integers_without_wrapping():
    res = vc.nested_pie(pd.DataFrame({"o": ["x", "x", "y"], "i": ["p", "p", "q"], "v": BIG}), "o", "i", "v")
    assert res.table["value"].tolist() == [2.0**63, 5.0]
    assert res.table["percent_of_total"].sum() == pytest.approx(100)


def test_nested_pie_percentages_survive_a_total_past_the_int64_range():
    v = np.full(3, 2**61 + 2**60, dtype=np.int64)  # each group fits, their total does not
    res = vc.nested_pie(pd.DataFrame({"o": ["x", "y", "z"], "i": ["p", "q", "r"], "v": v}), "o", "i", "v")
    assert res.table["percent_of_total"].tolist() == pytest.approx([100 / 3] * 3)
