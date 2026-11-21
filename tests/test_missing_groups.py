"""Rows whose group is missing: left out and counted by the inferential charts, drawn as "(missing)" by the others."""

import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets

GONE = [1, 2, 3, 44, 45]  # trial rows given a missing arm (and an outlying score) below


def _as(values, dtype):
    s = pd.Series(values, dtype=object)
    if dtype == "category":
        return s.astype("category")
    return s.astype(dtype)


@pytest.fixture(params=[object, "string", "category"])
def trial_gaps(request):
    t = datasets.trial()
    arm = t["arm"].astype(object)
    arm[GONE] = None
    t["arm"] = _as(arm, request.param)
    t.loc[GONE, "score"] = 500.0
    return t


# --- inferential charts leave the rows out, count them, and take threshold="mean" from the rows drawn ----------


@pytest.mark.parametrize("threshold", ["mean", "median"])
def test_benchmark_bar_threshold_comes_from_the_rows_drawn(trial_gaps, threshold):
    res = vc.benchmark_bar(trial_gaps, x="arm", y="score", threshold=threshold)
    kept = trial_gaps.dropna(subset=["arm"])
    same = vc.benchmark_bar(kept, x="arm", y="score", threshold=threshold)
    expected = kept["score"].mean() if threshold == "mean" else kept["score"].median()
    assert res.info["threshold"] == pytest.approx(expected) == same.info["threshold"]
    assert res.table["n"].sum() == len(kept) == 115
    assert res.table["status"].tolist() == same.table["status"].tolist() == ["indistinguishable"] * 3
    assert res.info["dropped_missing_group"] == 5 and same.info["dropped_missing_group"] == 0
    assert res.axes.lines[-1].get_ydata()[0] == res.info["threshold"]


@pytest.mark.parametrize("threshold", ["mean", "median"])
def test_centered_bar_threshold_comes_from_the_rows_drawn(trial_gaps, threshold):
    res = vc.centered_bar(trial_gaps, x="arm", y="score", threshold=threshold)
    kept = trial_gaps.dropna(subset=["arm"])
    same = vc.centered_bar(kept, x="arm", y="score", threshold=threshold)
    assert res.info["threshold"] == same.info["threshold"]
    assert res.table["n_above"].tolist() == same.table["n_above"].tolist()
    assert res.table["n_above"].sum() >= 50  # the outliers no longer lift the mean above almost every score
    assert res.info["dropped_missing_group"] == 5 and same.info["dropped_missing_group"] == 0
    assert res.axes.get_title(loc="left") == same.axes.get_title(loc="left")


@pytest.mark.parametrize("func", [vc.benchmark_bar, vc.centered_bar])
def test_threshold_mean_leaves_out_groups_not_in_order(func):
    df = pd.DataFrame({"g": list("aabbcc"), "y": [1.0, 3.0, 5.0, 7.0, 100.0, 200.0]})
    res = func(df, x="g", y="y", order=["a", "b"])
    assert res.info["threshold"] == 4.0
    assert res.info["dropped_missing_group"] == 0
    assert func(df, x="g", y="y", order=["a", "b"], threshold="median").info["threshold"] == 4.0


@pytest.mark.filterwarnings("ignore:All-NaN axis encountered:RuntimeWarning")  # Matplotlib 3.7 on a bar of NaN
@pytest.mark.filterwarnings("ignore:Warning. converting a masked element to nan:UserWarning")
@pytest.mark.parametrize("func", [vc.benchmark_bar, vc.centered_bar])
def test_threshold_mean_needs_a_value_in_a_group_drawn(func):
    df = pd.DataFrame({"g": ["a", "a", None, None], "y": [np.nan, np.nan, 1.0, 2.0]})
    with pytest.raises(ValueError, match=r"threshold='mean' needs a value of 'y' in one of the groups drawn"):
        func(df, x="g", y="y")
    res = func(df, x="g", y="y", threshold=1.5)  # a number needs no data
    assert res.table["n"].tolist() == [0] and res.info["dropped_missing_group"] == 2
    with pytest.raises(ValueError, match="column 'y' has no non-missing values"):
        func(df.assign(y=np.nan), x="g", y="y", threshold="nope")


def test_threshold_without_missing_groups_is_unchanged():
    trial = datasets.trial()
    assert vc.benchmark_bar(trial, x="arm", y="score").info["threshold"] == 51.565
    assert vc.centered_bar(trial, x="arm", y="score").info["threshold"] == 51.565
    assert vc.benchmark_bar(trial, x="arm", y="score").info["dropped_missing_group"] == 0


def test_estimation_plot_counts_the_rows_left_out(trial_gaps):
    res = vc.estimation_plot(trial_gaps, x="arm", y="score", n_resamples=200)
    same = vc.estimation_plot(trial_gaps.dropna(subset=["arm"]), x="arm", y="score", n_resamples=200)
    assert res.info["dropped_missing_group"] == 5 and same.info["dropped_missing_group"] == 0
    pd.testing.assert_frame_equal(res.table, same.table)
    pd.testing.assert_frame_equal(res.info["comparisons"], same.info["comparisons"])


def test_raincloud_counts_the_rows_left_out(trial_gaps):
    res = vc.raincloud(trial_gaps, x="arm", y="score")
    assert res.table["n"].tolist() == [37, 40, 38]
    assert res.info["dropped_missing_group"] == 5
    assert vc.raincloud(datasets.trial(), x="arm", y="score").info["dropped_missing_group"] == 0


def test_compare_correlations_counts_the_rows_left_out():
    m = datasets.measurements()
    m["group"] = m["group"].astype(object)
    m.loc[[0, 1, 2], "group"] = None
    res = vc.compare_correlations(m, group="group", columns=["length", "width", "depth"])
    assert res.info["dropped_missing_group"] == 3
    assert sorted(set(res.table["group_a"]) | set(res.table["group_b"])) == ["alpha", "beta", "gamma"]
    clean = vc.compare_correlations(datasets.measurements(), group="group", columns=["length", "width", "depth"])
    assert clean.info["dropped_missing_group"] == 0
