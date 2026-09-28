"""Empty columns, extreme magnitudes, reserved output names, level labels and bin checks."""

from __future__ import annotations

import re
from decimal import Decimal
from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets
from viz_calc import stats as st
from viz_calc._core import level_percent

FEATURES = ["length", "width", "depth", "mass"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --- all-missing columns ----------------------------------------------------------------------------------------

_EMPTY = pd.DataFrame({"g": ["a", "a", "b", "b"], "y": [np.nan] * 4, "z": [1.0, 2.0, 3.0, 4.0]})


@pytest.mark.parametrize("call", [
    lambda: vc.divergent_bar(_EMPTY, category="g", left="y", right="z"),
    lambda: vc.dumbbell(_EMPTY, label="g", start="y", end="z"),
    lambda: vc.likert(pd.DataFrame({"y": [np.nan] * 3}), items=["y"], levels=[1, 2, 3]),
    lambda: vc.correlogram(_EMPTY[["y", "z"]]),
    lambda: vc.correlogram(_EMPTY, columns=["z", "y"]),
    lambda: vc.estimation_plot(_EMPTY, x="g", y="y"),
    lambda: vc.lollipop(_EMPTY, "g", "y", stat="count"),
])
def test_all_missing_columns_are_rejected(call):
    with pytest.raises(ValueError, match="column 'y' has no non-missing values"):
        call()
    assert plt.get_fignums() == []


def test_divergent_bar_keeps_a_missing_sum_missing_rather_than_zero():
    d = pd.DataFrame({"g": ["a", "b", "b"], "y": [np.nan, 5.0, 1.0], "z": [3.0, 2.0, np.nan]})
    t = vc.divergent_bar(d, category="g", left="y", right="z").table
    assert np.isnan(t.loc[0, "y"]) and t.loc[1, "y"] == 6.0
    assert t["z"].tolist() == [3.0, 2.0]


def test_divergent_bar_sorts_without_overwriting_a_column_named_total():
    d = pd.DataFrame({"g": ["a", "b", "c"], "total": [1, 5, 2], "z": [3, 2, 0]})
    t = vc.divergent_bar(d, category="g", left="total", right="z", sort=True).table
    assert t.columns.tolist() == ["g", "total", "z"]
    assert t["g"].tolist() == ["c", "a", "b"] and t["total"].tolist() == [2, 1, 5]


def test_profile_boxes_leaves_out_an_empty_column():
    m = datasets.measurements().assign(notes=np.nan)
    t = vc.profile_boxes(m).table
    assert "notes" not in set(t["categorical"]) | set(t["numeric"])
    assert len(t) == len(vc.profile_boxes(datasets.measurements()).table)


def test_profile_boxes_draws_an_empty_panel_for_a_pair_without_complete_rows():
    d = pd.DataFrame({"g": ["a", "b"] * 10 + [None] * 20, "x": [np.nan] * 20 + list(range(20)),
                      "y": np.arange(40.0)})
    r = vc.profile_boxes(d, categorical=["g"], numeric=["x", "y"])
    t = r.table.set_index("numeric")
    assert t.loc["x", "n"] == 0 and np.isnan(t.loc["x", "eta_squared"])
    assert t.loc["y", "n"] == 20
    titles = [ax.get_title(loc="left") for ax in r.figure[0].axes]
    assert "x by g  (n=0)" in titles and any(t.startswith("y by g") for t in titles)


# --- magnitudes beyond 1e+-154 ----------------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e155, 1e-200])
def test_sd_columns_and_estimation_plot_at_extreme_magnitudes(scale):
    d = pd.DataFrame({"g": ["a"] * 3 + ["b"] * 3, "y": [v * scale for v in (1, 2, 3, 4, 5, 7)]})
    t = vc.benchmark_bar(d, "g", "y").table
    assert t["sd"].tolist() == pytest.approx([scale, np.sqrt(7 / 3) * scale], rel=1e-12)
    assert t["se"].tolist() == pytest.approx([scale / np.sqrt(3), np.sqrt(7 / 9) * scale], rel=1e-12)
    assert vc.histogram(d, "y").table.loc[0, "sd"] == pytest.approx(np.std([1, 2, 3, 4, 5, 7], ddof=1) * scale)
    for name, fn in (("ridgeplot", lambda: vc.ridgeplot(d, "y", "g")), ("raincloud", lambda: vc.raincloud(d, "g", "y"))):
        assert fn().table["sd"].tolist() == pytest.approx(t["sd"].tolist(), rel=1e-12), name
    r = vc.estimation_plot(d, "g", "y", n_resamples=200)
    assert r.table["sd"].tolist() == pytest.approx(t["sd"].tolist(), rel=1e-12)
    assert len(r.axes[1].collections) > 0  # the bootstrap density was drawn
    unit = vc.estimation_plot(d.assign(y=[1, 2, 3, 4, 5, 7]), "g", "y", n_resamples=200).info["comparisons"]
    cols = ["difference", "ci_low", "ci_high", "hedges_g", "p"]
    got = r.info["comparisons"][cols].to_numpy() / [scale, scale, scale, 1, 1]
    assert got == pytest.approx(unit[cols].to_numpy(), rel=0.05)  # BCa limits move with the last bits of the data


def test_benchmark_bar_sd_near_the_float_limit():
    t = vc.benchmark_bar(pd.DataFrame({"g": ["b", "b"], "y": [-1e308, 1e307]}), "g", "y", threshold=0).table
    assert t.loc[0, "sd"] == pytest.approx(7.778174593052023e307)
    assert t.loc[0, "se"] == pytest.approx(5.5e307)


def test_histogram_density_curve_at_a_tiny_scale():
    d = pd.DataFrame({"y": np.linspace(1, 2, 50) * 1e-200})
    r = vc.histogram(d, "y", stat="density", density_curve=True)
    curve = r.axes.flat[0].lines[0].get_ydata()
    assert np.isfinite(curve).all() and curve.max() == pytest.approx(1e200, rel=0.5)


# --- output names ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,call", [
    ("status", lambda d: vc.benchmark_bar(d, "status", "score")),
    ("n", lambda d: vc.benchmark_bar(d, "n", "score")),
    ("ci_low", lambda d: vc.estimation_plot(d, "ci_low", "score")),
    ("n", lambda d: vc.centered_bar(d, "n", "score")),
    ("p_above", lambda d: vc.centered_bar(d, "p_above", "score")),
    ("value", lambda d: vc.lollipop(d, "value", "score")),
    ("mean", lambda d: vc.ridgeplot(d, "score", "mean")),
    ("sd", lambda d: vc.raincloud(d, "sd", "score")),
])
def test_a_group_column_named_like_an_output_column_is_rejected(name, call):
    d = datasets.trial().rename(columns={"arm": name})
    with pytest.raises(ValueError, match=re.escape(f"column {name!r}: the table uses")):
        call(d)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("role", ["label", "start", "end"])
def test_dumbbell_rejects_columns_named_like_its_change_columns(role):
    names = {"label": "item", "start": "before", "end": "after", role: "change"}
    d = pd.DataFrame({names["label"]: ["a", "b"], names["start"]: [1.0, 2.0], names["end"]: [3.0, 1.0]})
    with pytest.raises(ValueError, match=re.escape(f"rename the {role} column 'change'")):
        vc.dumbbell(d, names["label"], names["start"], names["end"])


def test_pca_plot_and_radar_reject_group_names_taken_by_the_table():
    m = datasets.measurements()
    with pytest.raises(ValueError, match="rename the group column 'PC2'"):
        vc.pca_plot(m.rename(columns={"group": "PC2"}), FEATURES, group="PC2")
    vc.pca_plot(m.rename(columns={"group": "PC3"}), FEATURES, group="PC3")  # not plotted, so free
    with pytest.raises(ValueError, match="rename the group column 'mass_mean'"):
        vc.radar(m.rename(columns={"group": "mass_mean"}), FEATURES, group="mass_mean")


def test_group_levels_stay_in_tables_with_ordinary_names():
    t = vc.benchmark_bar(datasets.trial(), "arm", "score").table
    assert t["arm"].tolist() == ["control", "low dose", "high dose"]


# --- pca_plot table ---------------------------------------------------------------------------------------------


def test_pca_plot_with_a_multiindex_lists_index_tuples():
    m = datasets.measurements()
    m.index = pd.MultiIndex.from_arrays([np.arange(len(m)) % 3, np.arange(len(m))], names=["batch", "id"])
    t = vc.pca_plot(m, FEATURES, group="group").table
    assert t.columns.tolist() == ["row", "PC1", "PC2", "group"]
    assert t["row"].tolist()[:3] == [(0, 0), (1, 1), (2, 2)]
    assert t["group"].tolist() == m["group"].tolist()


def test_pca_plot_with_a_group_column_named_row():
    m = datasets.measurements()
    t = vc.pca_plot(m.rename(columns={"group": "row"}), FEATURES, group="row").table
    assert t.columns.tolist() == ["index", "PC1", "PC2", "row"]
    assert t["row"].tolist() == m["group"].tolist()
    assert t["index"].tolist() == m.index.tolist()


def test_pca_plot_table_keeps_plain_index_labels():
    m = datasets.measurements().iloc[::2]
    t = vc.pca_plot(m, FEATURES, group="group").table
    assert t["row"].tolist() == m.index.tolist() and t.index.tolist() == list(range(len(m)))


# --- confidence level labels --------------------------------------------------------------------------------------


@pytest.mark.parametrize("level,text", [(0.95, "95%"), (0.9, "90%"), (0.975, "97.5%"), (0.999, "99.9%"),
                                        (0.995, "99.5%"), (np.float32(0.95), "95%"), (Fraction(9, 10), "90%")])
def test_level_percent_keeps_every_digit(level, text):
    assert level_percent(level) == text


def test_chart_titles_show_the_level_unrounded():
    t, m = datasets.trial(), datasets.measurements()
    assert vc.benchmark_bar(t, "arm", "score", level=0.975).axes.get_title(loc="left").endswith("(bars: 97.5% CI)")
    assert "99.9% Wilson CI" in vc.centered_bar(t, "arm", "score", level=0.999).axes.get_title(loc="left")
    titles = [a.get_title(loc="left") for a in vc.estimation_plot(t, "arm", "score", level=0.975, n_resamples=200).axes]
    assert "97.5% CI" in titles[0] and "bootstrap 97.5% CI" in titles[1]
    assert "ellipses: 99.9% of data" in vc.pca_plot(m, FEATURES, group="group", level=0.999).axes.get_title(loc="left")


# --- level types ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("level", [Fraction(9, 10), Decimal("0.9"), np.array(0.9), np.float32(0.9)])
def test_levels_of_any_real_type_are_used_as_floats(level):
    assert st.mean_ci([1.0, 2, 3, 4, 5], level) == pytest.approx(st.mean_ci([1.0, 2, 3, 4, 5], 0.9))
    assert st._check_level(level) == pytest.approx(0.9) and type(st._check_level(level)) is float
    r = vc.benchmark_bar(datasets.trial(), "arm", "score", level=level)
    assert "90% CI" in r.axes.get_title(loc="left")


@pytest.mark.parametrize("level", ["0.9", True, np.array([0.9]), 1 + 0j, Decimal("NaN"), None])
def test_levels_that_are_not_real_numbers_are_rejected(level):
    with pytest.raises(ValueError, match="level must be strictly between 0 and 1"):
        st.mean_ci([1.0, 2, 3], level)


# --- histogram bins ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bins", [10.0, 0, -3, True, [1, 1, 2], [3, 2], [5], [[0, 1], [2, 3]], ["a", "b"],
                                  [0, np.inf], "doane"])
def test_histogram_rejects_bad_bins(bins):
    with pytest.raises(ValueError, match="bins must be one of"):
        vc.histogram(datasets.trial(), "score", bins=bins)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("bins", [8, np.int64(8), [0, 50, 100], np.array([0.0, 50, 100]), (0, 50, 100)])
def test_histogram_accepts_counts_and_edges(bins):
    assert vc.histogram(datasets.trial(), "score", bins=bins).info["bin_rule"] == "user"


def test_histogram_reports_values_outside_given_edges():
    r = vc.histogram(pd.DataFrame({"x": [1, 2, 3, 100]}), "x", bins=[0, 5], stat="percent")
    assert r.info["outside_bins"] == 1
    assert "1 value outside the bins not shown" in r.figure._suptitle.get_text()
    assert [p.get_height() for p in r.axes.flat[0].patches] == [100.0]  # over the values inside the edges
    assert r.table["n"].tolist() == [4]
    assert vc.histogram(pd.DataFrame({"x": [1, 2, 3, 100]}), "x").info["outside_bins"] == 0


# --- largest_remainder ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("values,total", [([64.0, 35.0], 8279144092626023), ([18.0, 36.0], 20826349215076146),
                                          ([1, 1, 1], 10**17), ([1, 2, 3], 2**70), ([1e308, 1e-300], 10**20)])
def test_largest_remainder_sums_to_large_totals(values, total):
    out = st.largest_remainder(values, total)
    assert sum(int(v) for v in out) == total


def test_largest_remainder_large_total_example():
    assert st.largest_remainder([1, 1, 1], 10**17).tolist() == [33333333333333334, 33333333333333333,
                                                                  33333333333333333]
    assert st.largest_remainder([1, 1, 1], 100).dtype.kind == "i"


# --- centered_bar labels and lollipop counts --------------------------------------------------------------------


def test_centered_bar_labels_never_round_to_all_or_none():
    d = pd.DataFrame({"g": ["a"] * 200 + ["b"] * 4, "y": [1] * 199 + [0] + [1, 1, 0, 0]})
    r = vc.centered_bar(d, "g", "y", threshold=0.5)
    assert [t.get_text() for t in r.axes.texts] == ["99.5%", "50%", "0.5%", "50%"]


def test_lollipop_count_counts_non_missing_values_of_y():
    df = pd.DataFrame({"x": ["a", "a", "b"], "y": [1, None, 2]})
    assert vc.lollipop(df, "x", "y", stat="count").table["value"].tolist() == [1, 1]
    assert sorted(vc.lollipop(df, "x", stat="count").table["value"].tolist()) == [1, 2]
