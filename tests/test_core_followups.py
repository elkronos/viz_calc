"""Whole-number FD limits, boolean column labels, datetime references and input checks."""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import stats as st
from viz_calc._core import check_count, select_columns


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --- Freedman-Diaconis on whole numbers -------------------------------------------------------------------------


@pytest.mark.parametrize("outlier", [8000, 9000, 50_000])
def test_whole_number_fd_limit_uses_the_rounded_bin_count(outlier):
    counts = np.random.default_rng(0).poisson(2, 100_000)
    edges, rule = st._bin_edges(np.r_[counts, outlier], "fd")
    assert rule == "fd (whole-number widths for integer data)"
    assert len(edges) - 1 == outlier + 1
    assert np.all(np.diff(edges) == 1.0)


def test_whole_number_fd_still_falls_back_when_the_rounded_count_is_too_large():
    counts = np.random.default_rng(0).poisson(2, 100_000)
    edges, rule = st._bin_edges(np.r_[counts, 200_000], "fd")
    assert rule == "sturges (FD would need 200,001 bins)"


def test_fractional_fd_limit_is_unchanged():
    x = np.r_[np.random.default_rng(0).normal(size=100_000), 1e6]
    _, rule = st._bin_edges(x, "fd")
    assert rule.startswith("sturges (FD would need")


# --- columns labelled False / True ------------------------------------------------------------------------------


def _bool_frame():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({False: rng.normal(size=40), True: rng.normal(size=40), "g": rng.choice(["a", "b"], 40)})
    d[True] = d[True] + d[False]
    return d


def test_select_columns_reads_booleans_as_labels_in_the_given_order():
    d = _bool_frame()
    assert select_columns(d, [True, False]).columns.tolist() == [True, False]
    assert select_columns(d, ["g", True]).columns.tolist() == ["g", True]
    with pytest.raises(KeyError):
        select_columns(d, ["missing"])


def test_divergent_bar_with_boolean_column_labels():
    f = pd.DataFrame({"t": ["b", "a", "b"], False: [1, 2, 3], True: [3, 4, 5]})
    t = vc.divergent_bar(f, "t", True, False).table
    assert t.columns.tolist() == ["t", True, False]
    assert t.set_index("t").loc["b"].tolist() == [8, 4]
    with pytest.raises(ValueError, match="non-negative"):
        vc.divergent_bar(f.assign(**{"t": ["a", "b", "c"]}).replace({4: -4}), "t", False, True)


def test_likert_with_boolean_levels():
    yn = pd.DataFrame({"q1": [True, True, False, True], "q2": [False, False, True, False]})
    t = vc.likert(yn, ["q1", "q2"], [False, True]).table.set_index("item")
    assert t.loc["q1", True] == 75.0 and t.loc["q1", False] == 25.0
    assert t.loc["q1", "net"] == 50.0 and t.loc["q2", "net"] == -50.0


def test_pca_and_pca_plot_with_boolean_feature_labels():
    d = _bool_frame()
    res = vc.pca(d, [False, True])
    assert res["loadings"].index.tolist() == [False, True]
    r = vc.pca_plot(d, [True, False], group="g", loadings=True)
    assert sorted(t.get_text() for t in r.axes.texts if t.get_text()) == ["False", "True"]
    assert len(r.table) == len(d)


def test_profile_scatters_boxes_and_quadrant_with_boolean_labels():
    d = _bool_frame()
    t = vc.profile_scatters(d.drop(columns="g")).table
    assert len(t) == 1 and t.loc[0, "r"] == pytest.approx(np.corrcoef(d[False], d[True])[0, 1])
    b = pd.DataFrame({False: np.repeat(["x", "y"], 20), True: np.arange(40.0)})
    assert vc.profile_boxes(b, categorical=[False], numeric=[True]).table.loc[0, "n"] == 40
    assert vc.quadrant_plot(d, True, False).table["n"].sum() == 40


# --- estimation_plot with a datetime reference ------------------------------------------------------------------


def _dated():
    rng = np.random.default_rng(0)
    days = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
    return pd.DataFrame({"d": np.repeat(days, 10), "y": rng.normal(size=30)})


@pytest.mark.parametrize("kwargs", [
    {"reference": np.datetime64("2024-02-01")},
    {"reference": pd.Timestamp("2024-02-01"),
     "order": np.array(["2024-01-01", "2024-02-01", "2024-03-01"], dtype="datetime64[D]")},
    {"reference": pd.Timestamp("2024-02-01")},
])
def test_estimation_plot_accepts_a_datetime_reference_of_either_type(kwargs):
    r = vc.estimation_plot(_dated(), "d", "y", n_resamples=200, **kwargs)
    assert r.axes[1].get_ylabel() == "Difference from\n2024-02-01"
    comps = r.info["comparisons"]
    assert len(comps) == 2
    assert all(pd.Timestamp(g) != pd.Timestamp("2024-02-01") for g in comps["group"])


# --- repeated columns -------------------------------------------------------------------------------------------


def test_correlogram_rejects_a_repeated_column():
    m = vc.datasets.measurements()
    with pytest.raises(ValueError, match=re.escape("columns repeats column(s): ['length']")):
        vc.correlogram(m, columns=["length", "length", "width"])
    assert plt.get_fignums() == []


def test_pca_and_likert_reject_a_repeated_column():
    d = _bool_frame()
    with pytest.raises(ValueError, match="features repeats column"):
        vc.pca(d, [False, True, False])
    with pytest.raises(ValueError, match="items repeats column"):
        vc.likert(pd.DataFrame({"q1": ["a", "b"]}), ["q1", "q1"], ["a", "b"])


# --- likert level names -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("levels", [["n", "y"], ["no", "net", "yes"], ["item", "other"]])
def test_likert_rejects_levels_named_like_its_table_columns(levels):
    data = pd.DataFrame({"q1": [levels[0], levels[-1]]})
    with pytest.raises(ValueError, match=r"rename level\(s\)"):
        vc.likert(data, ["q1"], levels)
    assert plt.get_fignums() == []


# --- largest_remainder total ------------------------------------------------------------------------------------


@pytest.mark.parametrize("total", [2.5, -3, 2.0, True, "3", None])
def test_largest_remainder_rejects_a_total_that_is_not_a_non_negative_integer(total):
    with pytest.raises(ValueError, match=re.escape(f"total must be a non-negative integer, got {total!r}")):
        st.largest_remainder([1, 2], total)


@pytest.mark.parametrize("total", [0, 3, np.int64(3)])
def test_largest_remainder_accepts_integer_totals(total):
    out = st.largest_remainder([1, 2], total)
    assert out.sum() == total


# --- count checks -----------------------------------------------------------------------------------------------


def test_check_count_explains_that_a_whole_float_must_be_an_int():
    with pytest.raises(ValueError, match=re.escape("col_wrap must be a whole number of at least 1 given as an int, "
                                                   "got the float 2.0; use 2")):
        check_count("col_wrap", 2.0)
    with pytest.raises(ValueError, match=re.escape("ncols must be a whole number of at least 1, got 0.0")):
        check_count("ncols", 0.0)
    with pytest.raises(ValueError, match=re.escape("ncols must be a whole number of at least 1, got 2.5")):
        check_count("ncols", 2.5)
    check_count("ncols", np.int64(2))
