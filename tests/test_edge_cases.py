"""Edge cases and regressions found in the adversarial review of 1.0.0."""

import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets
from viz_calc import stats as st

# --- statistics --------------------------------------------------------------------------------------------------


def test_fd_bins_fall_back_to_sturges_when_iqr_is_zero():
    x = np.r_[np.zeros(80), np.arange(1, 21)]  # zero-inflated: IQR = 0
    edges = st.histogram_bins(x)
    assert len(edges) - 1 == len(np.histogram_bin_edges(x, bins="sturges")) - 1 > 1
    res = vc.histogram(pd.DataFrame({"x": x}), x="x")
    assert len(res.info["bin_edges"]) > 2
    assert res.info["bin_rule"].startswith("sturges")
    assert vc.histogram(datasets.trial(), x="score").info["bin_rule"] == "fd"


def test_histogram_bins_rejects_unknown_rule():
    with pytest.raises(ValueError, match="rule"):
        st.histogram_bins([1, 2, 3], "freedman")


@pytest.mark.parametrize("func", [lambda: st.correlation_test([1, 2, 3, 4], [1, 3, 2, 4], "kendall"),
                                  lambda: st.compare_correlations_test(0.5, 50, 0.3, 50, "kendall")])
def test_unsupported_correlation_method_raises(func):
    with pytest.raises(ValueError, match="method"):
        func()


def test_r_to_z_undefined_for_three_or_fewer():
    z, p = st.compare_correlations_test(0.99, 3, -0.99, 50)
    assert np.isnan(z) and np.isnan(p)
    z, p = st.compare_correlations_test(np.array([0.5, 0.5]), np.array([3, 40]), 0.1, 40)
    assert np.isnan(z[0]) and not np.isnan(z[1])


# --- validation --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("call", [
    lambda d: vc.benchmark_bar(d, x="g", y="v"),
    lambda d: vc.lollipop(d, x="g", y="v"),
    lambda d: vc.funnel(d, stage="g", value="v"),
    lambda d: vc.waterfall(d, label="g", value="v"),
])
def test_empty_data_gives_a_clear_error(call):
    with pytest.raises(ValueError, match="no rows"):
        call(pd.DataFrame({"g": pd.Series([], dtype=object), "v": pd.Series([], dtype=float)}))


def test_all_missing_grouping_column_gives_a_clear_error():
    with pytest.raises(ValueError, match="no non-missing values"):
        vc.benchmark_bar(pd.DataFrame({"g": [np.nan, np.nan], "v": [1.0, 2.0]}), x="g", y="v")


def test_quadrant_plot_rejects_unknown_method_and_constant_column():
    trial = datasets.trial()
    with pytest.raises(ValueError, match="method"):
        vc.quadrant_plot(trial, x="baseline", y="score", method="kendall")
    with pytest.raises(ValueError, match="single value"):
        vc.quadrant_plot(trial.assign(c=1.0), x="c", y="score")
    with pytest.raises(ValueError, match="method"):
        vc.profile_scatters(datasets.measurements(), method="kendall")


# --- composition -------------------------------------------------------------------------------------------------


def test_upset_dataframe_treats_missing_as_non_member():
    res = vc.upset(pd.DataFrame({"A": [1, np.nan, np.nan, 1], "B": [np.nan, 1, np.nan, 1]}))
    sizes = {(r.A, r.B): r.size for r in res.table.itertuples()}
    assert sizes == {(True, False): 1, (False, True): 1, (True, True): 1}
    assert res.info["set_sizes"].to_dict() == {"A": 2, "B": 2}


def test_upset_rejects_non_boolean_membership_and_bad_sort():
    with pytest.raises(ValueError, match="True/False"):
        vc.upset(pd.DataFrame({"A": ["1", "0"], "B": ["0", "1"]}))
    with pytest.raises(ValueError, match="sort_by"):
        vc.upset({"A": {1}, "B": {2}}, sort_by="name")


def test_upset_degree_order_keeps_largest_intersections():
    sets = {"A": set(range(0, 50)) | {100}, "B": set(range(40, 60)) | {100}, "C": {100, 200}}
    res = vc.upset(sets, max_intersections=2, sort_by="degree")
    assert sorted(res.table["size"], reverse=True) == [40, 10]


def test_waffle_order_must_be_complete():
    survey = datasets.survey()
    with pytest.raises(ValueError, match="not in the data"):
        vc.waffle(survey, category="team", order=["Data", "Sales", "Engneering"])
    with pytest.raises(ValueError, match="must list every level"):
        vc.waffle(survey, category="team", order=["Data", "Sales"])
    res = vc.waffle(survey, category="team", order=["Data", "Sales", "Engineering", "Design"])
    assert list(res.table["team"]) == ["Data", "Sales", "Engineering", "Design"]
    assert res.table.set_index("team").loc["Data", "percent"] == pytest.approx(22.0)


def test_waffle_rejects_all_zero():
    with pytest.raises(ValueError, match="positive total"):
        vc.waffle(pd.DataFrame({"c": ["a", "b"], "v": [0, 0]}), category="c", value="v")


def test_stacked_percentages_category_order_must_be_complete():
    survey = datasets.survey()
    with pytest.raises(ValueError, match="must list every level"):
        vc.stacked_percentages(survey, group="team", category="Meetings are useful",
                               category_order=["Agree", "Strongly agree"])


def test_stacked_percentages_group_with_only_missing_categories():
    # crashed on pandas 2.0 (stack() dropped NaN percentages but not the counts)
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "c": ["x", "y", np.nan, np.nan]})
    t = vc.stacked_percentages(df, group="g", category="c").table
    assert t.loc[t.g == "a", "percent"].tolist() == [50, 50]
    assert t.loc[t.g == "b", "n"].sum() == 0 and t.loc[t.g == "b", "percent"].sum() == 0


def test_percent_grid_rejects_success_value_that_never_occurs():
    df = pd.DataFrame({"a": ["yes", "no", "yes"]})
    with pytest.raises(ValueError, match="does not occur"):
        vc.percent_grid(df, column="a", success="Yes")
    assert vc.percent_grid(pd.DataFrame({"a": [0, 0, 0]}), column="a").table["percent"].iloc[0] == 0


# --- correlation, multivariate, network --------------------------------------------------------------------------


def test_correlogram_hidden_cells_keep_their_values_in_info():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=30), "b": rng.normal(size=30)})
    df["c"] = df.a + rng.normal(scale=0.3, size=30)
    res = vc.correlogram(df, hide_nonsignificant=True)
    r_ab = res.table.set_index(["var1", "var2"]).loc[("a", "b"), "r"]
    assert r_ab != 0 and res.info["matrix"].loc["a", "b"] == pytest.approx(r_ab)


def test_radar_leaves_missing_group_metric_as_gap():
    m = datasets.measurements()
    m.loc[m.group == "alpha", "mass"] = np.nan
    t = vc.radar(m, metrics=["length", "width", "depth", "mass"], group="group").table.set_index("group")
    assert np.isnan(t.loc["alpha", "mass_scaled"])
    flat = vc.radar(m.assign(mass=1.0), metrics=["length", "width", "mass"], group="group").table
    assert (flat["mass_scaled"] == 0.5).all()


def test_pca_plot_with_duplicate_index_labels():
    m = datasets.measurements()
    m.index = [0] * len(m)
    res = vc.pca_plot(m, features=["length", "width", "depth", "mass"], group="group")
    assert list(res.table["group"].value_counts().sort_index()) == [50, 50, 50]


def test_network_merges_repeated_edges():
    e = pd.DataFrame({"source": ["A", "A", "B", "C"], "target": ["B", "B", "A", "A"], "w": [5, 3, 4, 1]})
    res = vc.network_map(e, "source", "target", weight="w")
    assert res.table.set_index("node")["strength"].to_dict() == {"A": 13, "B": 12, "C": 1}
    assert res.info["duplicate_rows_merged"] == 2
    directed = vc.network_map(e, "source", "target", weight="w", directed=True)
    assert directed.info["graph"]["A"]["B"]["w"] == 8 and directed.info["graph"]["B"]["A"]["w"] == 4
    with pytest.raises(ValueError, match="non-negative"):
        vc.network_map(e.assign(w=-1), "source", "target", weight="w")


# --- time --------------------------------------------------------------------------------------------------------


def test_calendar_heatmap_timezone_aware_dates():
    d = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10, tz="UTC"), "v": range(10)})
    res = vc.calendar_heatmap(d, date="date", value="v")
    drawn = ~np.ma.getmaskarray(res.axes[0].collections[0].get_array())
    assert drawn.sum() == 10


def test_calendar_heatmap_all_missing_day_is_not_zero():
    d = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02"]), "v": [np.nan, 1.0, 2.0]})
    t = vc.calendar_heatmap(d, date="date", value="v").table.set_index("date")["value"]
    assert np.isnan(t.iloc[0]) and t.iloc[1] == 3


def test_timeseries_fill_parses_string_dates_before_sorting():
    ts = pd.DataFrame({"t": ["1/2/2024", "1/10/2024", "1/3/2024", "1/20/2024", "2/1/2024"],
                       "a": [1, 3, 2, 5, 6], "b": [2] * 5})
    res = vc.timeseries_fill(ts, time="t", series=["a", "b"])
    assert res.info["crossovers"] == 1
    assert res.table["t"].is_monotonic_increasing


def test_animated_bubble_ignores_missing_values_for_limits():
    df = pd.DataFrame({"t": [1, 1, 2, 2], "x": [0, np.nan, 0, 1], "y": [0, 1, np.nan, 0], "s": [1, 2, 3, 4]})
    res = vc.animated_bubble(df, time="t", x="x", y="y", size="s")
    assert np.isfinite(res.axes.get_xlim()).all() and np.isfinite(res.axes.get_ylim()).all()


def test_gantt_with_missing_group():
    proj = datasets.projects()
    proj.loc[0, "team"] = np.nan
    res = vc.gantt(proj, task="task", start="start", end="end", group="team")
    assert len(res.table) == len(proj)


# --- compare -----------------------------------------------------------------------------------------------------


def test_dumbbell_accepts_boolean_columns():
    df = pd.DataFrame({"k": ["a", "b"], "before": [True, False], "after": [True, True]})
    assert vc.dumbbell(df, label="k", start="before", end="after").table["change"].tolist() == [0, 1]


def test_benchmark_group_without_values_is_no_data():
    df = pd.DataFrame({"g": ["a", "a", "a", "b"], "v": [1.0, 2.0, 3.0, np.nan]})
    status = vc.benchmark_bar(df, x="g", y="v", threshold=2, classify="mean").table.set_index("g")["status"]
    assert status["b"] == "no data"
