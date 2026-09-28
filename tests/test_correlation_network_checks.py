"""Correlation and network charts: input validation, level checks and figure cleanup."""

import inspect
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import compare, correlation, datasets, network


@pytest.fixture(scope="module")
def trial():
    return datasets.trial()


@pytest.fixture(scope="module")
def edges():
    return pd.DataFrame({"s": ["a", "b", "c", "a"], "t": ["b", "c", "a", "d"], "w": [1.0, 2.0, 3.0, 4.0]})


def _untouched(ax):
    return not (ax.lines or ax.patches or ax.collections or ax.images or ax.texts)


# --- figures opened by a call that fails are closed ------------------------------------------------------------


@pytest.mark.parametrize("module", [correlation, network])
def test_correlation_and_network_functions_clean_up_on_error(module):
    for name in module.__all__:
        func = getattr(module, name)
        assert hasattr(func, "__wrapped__"), name
        assert inspect.signature(func) == inspect.signature(func.__wrapped__)


def test_a_correlation_chart_that_fails_while_drawing_leaves_no_figure(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("drawing failed")

    m = datasets.measurements()
    monkeypatch.setattr(correlation, "_draw_matrix", broken)
    with pytest.raises(RuntimeError, match="drawing failed"):
        vc.correlogram(m)
    with pytest.raises(RuntimeError, match="drawing failed"):
        vc.compare_correlations(m, "group")
    monkeypatch.undo()
    monkeypatch.setattr(correlation.st, "correlation_test", broken)  # quadrant_plot tests after drawing
    with pytest.raises(RuntimeError, match="drawing failed"):
        vc.quadrant_plot(m, "length", "width")
    assert plt.get_fignums() == []


def test_a_network_map_that_fails_while_drawing_leaves_no_figure(edges, monkeypatch):
    nx = pytest.importorskip("networkx")

    def broken(*args, **kwargs):
        raise RuntimeError("labels failed")

    monkeypatch.setattr(nx, "draw_networkx_labels", broken)
    with pytest.raises(RuntimeError, match="labels failed"):
        vc.network_map(edges, "s", "t")
    assert plt.get_fignums() == []


# --- network_map options and empty columns -----------------------------------------------------------------------


@pytest.mark.parametrize("interactive", [False, True])
def test_network_map_rejects_an_unknown_size_by_before_drawing(edges, interactive):
    pytest.importorskip("networkx")
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match=r"size_by must be one of \['degree', 'strength', 'betweenness'\], "
                                         r"got 'betweeness'"):
        vc.network_map(edges, "s", "t", size_by="betweeness", interactive=interactive, ax=ax)
    assert _untouched(ax) and len(plt.get_fignums()) == 1


@pytest.mark.parametrize("change,weight,match", [
    ({"s": np.nan}, None, "column 's' has no non-missing values"),
    ({"t": None}, None, "column 't' has no non-missing values"),
    ({"w": np.nan}, "w", "column 'w' has no non-missing values"),
    ({"w": pd.array([None] * 4, dtype="Float64")}, "w", "column 'w' has no non-missing values"),
    ({"s": ["a", None, "c", None], "t": [None, "b", None, "d"]}, None, "no row has both a 's' and a 't' value"),
])
def test_network_map_explains_columns_without_edges(edges, change, weight, match):
    pytest.importorskip("networkx")  # these used to fail with "zero-size array to reduction operation minimum"
    with pytest.raises(ValueError, match=match):
        vc.network_map(edges.assign(**change), "s", "t", weight=weight)
    assert plt.get_fignums() == []


def test_network_map_still_drops_rows_missing_an_end(edges):
    pytest.importorskip("networkx")
    res = vc.network_map(edges.assign(t=["b", "c", None, "d"]), "s", "t", weight="w")
    assert sorted(res.table["node"]) == ["a", "b", "c", "d"] and res.info["graph"].number_of_edges() == 3


@pytest.mark.parametrize("change,match", [
    ({"w": np.nan}, "column 'w' has no non-missing values"),  # used to draw an empty diagram with zero flows
    ({"s": None}, "column 's' has no non-missing values"),
])
def test_sankey_rejects_columns_with_no_values(edges, change, match):
    pytest.importorskip("plotly")
    with pytest.raises(ValueError, match=match):
        vc.sankey(edges.assign(**change), "s", "t", "w")


# --- column lists --------------------------------------------------------------------------------------------------


def _corr_frame():
    rng = np.random.default_rng(4)
    d = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "ab"])
    return d.assign(g=np.repeat(["x", "y"], 20))


@pytest.mark.parametrize("call", [
    lambda d, cols: vc.correlogram(d, columns=cols),
    lambda d, cols: vc.compare_correlations(d, "g", columns=cols),
])
def test_correlation_column_lists_take_any_sequence_but_not_a_string(call):
    d = _corr_frame()
    with pytest.raises(TypeError, match=r"columns must be a list of column names, not a str; use \['ab'\]"):
        call(d, "ab")  # used to pick the single-letter columns 'a' and 'b'
    expected = call(d, ["b", "ab"]).table
    for cols in (pd.Index(["b", "ab"]), np.array(["b", "ab"]), ("b", "ab"), (c for c in ["b", "ab"])):
        pd.testing.assert_frame_equal(call(d, cols).table, expected)


# --- level is checked before anything is drawn ---------------------------------------------------------------------


@pytest.mark.parametrize("call", [
    lambda d, ax, lv: vc.benchmark_bar(d, x="arm", y="score", level=lv, ax=ax),
    lambda d, ax, lv: vc.centered_bar(d, x="arm", y="score", level=lv, ax=ax),
    lambda d, ax, lv: vc.pca_plot(datasets.measurements(), ["length", "width", "depth"], group="group", level=lv, ax=ax),
    lambda d, ax, lv: vc.pca_plot(datasets.measurements(), ["length", "width", "depth"], ellipse=None, level=lv, ax=ax),
])
@pytest.mark.parametrize("level", [95, 0, 1.0, -0.5, True, "0.95", None])
def test_level_is_checked_before_drawing(trial, call, level):
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="level must be strictly between 0 and 1"):
        call(trial, ax, level)
    assert _untouched(ax)


def test_estimation_plot_checks_level_before_opening_a_figure(trial, monkeypatch):
    def no_figure(*args, **kwargs):
        raise AssertionError("a figure was opened")

    monkeypatch.setattr(compare.plt, "subplots", no_figure)
    with pytest.raises(ValueError, match="level must be strictly between 0 and 1"):
        vc.estimation_plot(trial, x="arm", y="score", level=95)


def test_pca_plot_ellipse_follows_level():
    # level=95 used to reach chi2.ppf unchecked and draw NaN-sized ellipses
    m = datasets.measurements()
    feats = ["length", "width", "depth"]
    wide = vc.pca_plot(m, feats, group="group", level=0.95).info["ellipses"]
    narrow = vc.pca_plot(m, feats, group="group", level=np.float32(0.5)).info["ellipses"]
    for g, e in wide.items():
        assert np.isfinite(e["width"]) and narrow[g]["width"] < e["width"]


# --- docstrings match the behaviour --------------------------------------------------------------------------------


def _doc(func):
    return " ".join(inspect.getdoc(func).split())


def test_histogram_docstring_describes_whole_number_bins():
    counts = np.random.default_rng(0).poisson(3, 500)
    res = vc.histogram(pd.DataFrame({"visits": counts}), x="visits")
    assert np.all(np.diff(res.info["bin_edges"]) % 1 == 0) and np.all(res.info["bin_edges"] % 1 == 0.5)
    assert f'``info["bin_rule"]`` reads ``"{res.info["bin_rule"]}"``' in _doc(vc.histogram)
    assert "half-integers" in _doc(vc.histogram)


def test_profile_scatters_uses_exact_small_n_spearman_p_values():
    d = pd.DataFrame({"x": [1.0, 2, 3, 4, 5], "y": [2.0, 4, 5, 7, 9], "z": [5.0, 1, 4, 2, 3]})
    table = vc.profile_scatters(d, columns=["x", "y", "z"], method="spearman", p_adjust="none").table
    perfect = table[(table["x"] == "x") & (table["y"] == "y")].iloc[0]
    assert perfect["r"] == pytest.approx(1)
    assert perfect["p"] == pytest.approx(2 / math.factorial(5))  # the t approximation gives 0
    assert "exact permutation p-values" in _doc(vc.profile_scatters)
    with pytest.raises(ValueError, match=r"method must be one of \['pearson', 'spearman'\], got 'kendall'"):
        vc.profile_scatters(d, columns=["x", "y"], method="kendall")


# --- quadrant ties and date titles ---------------------------------------------------------------------------------


@pytest.mark.parametrize("standardize", [True, False])
def test_quadrant_point_on_a_mean_near_zero_counts_as_high(standardize):
    # The decimal mean of x is exactly 0, but the stored values sum to 2.8e-17, so 0.0 fell just below the mean;
    # a tolerance relative to the (near-zero) mean could not absorb that.
    df = pd.DataFrame({"x": [0.1, 0.2, -0.3, 0.0, 0.5, -0.5], "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    table = vc.quadrant_plot(df, "x", "y", standardize=standardize).table
    assert dict(zip(table["quadrant"], table["n"])) == {"high x, high y": 2, "low x, high y": 1,
                                                        "low x, low y": 1, "high x, low y": 2}


def test_quadrant_median_and_real_differences_are_not_widened():
    df = pd.DataFrame({"x": [-1.0, 0.0, 1e-17, 1.0], "y": [1.0, 2.0, 3.0, 4.0]})
    table = vc.quadrant_plot(df, "x", "y", center="median").table  # median 5e-18: 0.0 is really below it
    assert dict(zip(table["quadrant"], table["n"]))["low x, low y"] == 2
    df = pd.DataFrame({"x": [0.1, 0.2, -0.3, -1e-3, 0.5, -0.5], "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    table = vc.quadrant_plot(df, "x", "y").table
    assert dict(zip(table["quadrant"], table["n"]))["low x, high y"] == 2


def test_compare_correlations_titles_date_groups_as_dates():
    d = _corr_frame().assign(g=pd.to_datetime(np.repeat(["2024-01-01", "2024-02-01"], 20)))
    ax = vc.compare_correlations(d, "g").axes.flat[0]
    assert ax.get_title(loc="left") == "r(2024-02-01) − r(2024-01-01)"
