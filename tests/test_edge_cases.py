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
    with pytest.raises(ValueError, match="typo.*Engneering"):
        vc.waffle(survey, category="team", order=["Data", "Sales", "Engneering", "Design"])
    with pytest.raises(ValueError, match="must list every level"):
        vc.waffle(survey, category="team", order=["Data", "Sales"])
    with pytest.raises(ValueError, match="repeats"):
        vc.waffle(survey, category="team", order=["Data", "Data", "Sales", "Engineering", "Design"])
    res = vc.waffle(survey, category="team", order=["Data", "Sales", "Engineering", "Design"])
    assert list(res.table["team"]) == ["Data", "Sales", "Engineering", "Design"]
    assert res.table.set_index("team").loc["Data", "percent"] == pytest.approx(22.0)
    # an unused level of a response scale may be listed; it gets zero tiles
    res = vc.waffle(survey, category="team", order=["Data", "Sales", "Engineering", "Design", "Legal"])
    assert res.table.set_index("team").loc["Legal", "tiles"] == 0 and res.table["tiles"].sum() == 100


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
    assert t.loc[t.g == "b", "n"].sum() == 0 and t.loc[t.g == "b", "percent"].isna().all()


def test_stacked_percentages_deprecated_alias_and_reserved_names():
    survey = datasets.survey()
    levels = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
    with pytest.warns(FutureWarning, match="category_order_"):
        old = vc.stacked_percentages(survey, group="team", category="Meetings are useful", category_order_=levels)
    new = vc.stacked_percentages(survey, group="team", category="Meetings are useful", category_order=levels)
    pd.testing.assert_frame_equal(old.table, new.table)
    with pytest.raises(ValueError, match="repeats"):
        vc.stacked_percentages(survey, group="team", category="remote", group_order=["Data", "Data"])
    with pytest.raises(ValueError, match="output columns"):
        vc.stacked_percentages(survey.rename(columns={"team": "n"}), group="n", category="remote")


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


# --- second verification round -----------------------------------------------------------------------------------


@pytest.mark.parametrize("x", [
    np.r_[np.zeros(75), np.geomspace(1e-6, 1000, 25)],                       # IQR tiny but not zero
    np.r_[np.full(50, 1.0), np.full(45, 0.1 * 3 / 0.3), [0.5, 2, 3, 4, 5]],  # ties that differ by float noise
])
def test_fd_bins_fall_back_when_iqr_is_near_zero(x):
    for rule in ("fd", "auto"):
        edges, used = st._bin_edges(x, rule)
        assert 2 < len(edges) < 50 and used.startswith("sturges")


def test_largest_remainder_rejects_infinite_values():
    with pytest.raises(ValueError, match="finite"):
        st.largest_remainder([np.inf, 1, 2], 100)


def test_abbreviate_keeps_small_values_and_ticks_distinct():
    from viz_calc._core import abbreviate

    assert [abbreviate(v) for v in (0.034, 0.0021, -0.5, 0, 1105, 12345)] == ["0.034", "0.0021", "-0.5", "0", "1,105", "12.3K"]
    res = vc.lollipop(pd.DataFrame({"c": list("abcd"), "v": [0.034, 0.021, 0.012, 0.029]}), x="c", y="v")
    res.figure.canvas.draw()
    ticks = [t.get_text() for t in res.axes.get_xticklabels()]
    assert len(set(ticks)) == len(ticks)
    labels = [t.get_text() for t in res.axes.texts]
    assert labels == ["0.034", "0.029", "0.021", "0.012"]


def test_abbreviated_axis_has_no_offset_text():
    res = vc.divergent_bar(pd.DataFrame({"k": ["a", "b"], "l": [30000, 45000], "r": [26000, 52000]}),
                           category="k", left="l", right="r")
    res.figure.canvas.draw()
    ticks = [t.get_text() for t in res.axes.get_xticklabels()]
    assert "40K" in ticks and not any(t.startswith(("-", "−")) for t in ticks)
    assert res.axes.xaxis.get_major_formatter().get_offset() == ""


def test_to_pptx_keeps_aspect_ratio_and_titles_per_item(tmp_path):
    pptx = pytest.importorskip("pptx")
    meas = datasets.measurements()
    boxes = vc.profile_boxes(meas, per_page=4)  # 8 panels -> 2 pages
    radar = vc.radar(meas, metrics=["length", "width", "depth"], group="group")
    out = vc.to_pptx([vc.profile_bars(meas), boxes, radar], str(tmp_path / "d.pptx"), titles=["Counts", "Boxes", "Radar"])
    prs = pptx.Presentation(out)
    titles = [next((sh.text_frame.text for sh in sl.shapes if sh.has_text_frame), None) for sl in prs.slides]
    assert titles == ["Counts", "Boxes (1/2)", "Boxes (2/2)", "Radar"]
    from io import BytesIO

    from PIL import Image
    pic = [sh for sh in prs.slides[3].shapes if sh.shape_type == 13][0]
    w, h = Image.open(BytesIO(pic.image.blob)).size
    assert pic.width / pic.height == pytest.approx(w / h, rel=0.01)


def test_to_pptx_rejects_plotly(tmp_path):
    pytest.importorskip("pptx")
    pytest.importorskip("plotly")
    res = vc.sankey(pd.DataFrame({"a": ["x"], "b": ["y"], "v": [1]}), source="a", target="b", value="v")
    with pytest.raises(TypeError, match="Matplotlib"):
        vc.to_pptx(res, str(tmp_path / "p.pptx"))


def test_save_without_extension_returns_real_path(tmp_path):
    path = vc.waffle(datasets.survey(), category="team").save(str(tmp_path / "chart"))
    assert path.endswith(".png") and (tmp_path / "chart.png").exists()


def test_raincloud_draws_narrow_groups():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": ["fast"] * 100 + ["slow"] * 100,
                       "t": np.r_[rng.uniform(245, 254, 100), rng.uniform(250, 2250, 100)]})
    res = vc.raincloud(df, x="g", y="t")
    fills = [c for c in res.axes.collections if c.__class__.__name__ in ("PolyCollection", "FillBetweenPolyCollection")]
    assert len(fills) == 2


def test_dates_with_mixed_offsets_and_formats():
    d = pd.DataFrame({"when": ["2024-03-09 22:00:00-05:00", "2024-03-10 12:00:00-04:00", "2024-03-11 02:00:00-04:00"],
                      "v": [2.0, 23.0, 3.0]})
    t = vc.calendar_heatmap(d, date="when", value="v").table
    assert [str(x.date()) for x in t["date"]] == ["2024-03-09", "2024-03-10", "2024-03-11"]
    assert vc.period_bars(d, date="when", value="v", freq="day").table["value"].tolist() == [2, 23, 3]
    mixed = pd.DataFrame({"when": ["2024-01-01", "2024-01-02 10:30:00"], "v": [1.0, 2.0]})
    assert len(vc.calendar_heatmap(mixed, date="when", value="v").table) == 2


def test_timeseries_fill_keeps_object_years_numeric_and_marks_missing_leader():
    df = pd.DataFrame({"year": pd.Series([2021, 2022, 2023], dtype=object), "a": [1.0, np.nan, 3.0], "b": [2.0, 2.0, 2.0]})
    res = vc.timeseries_fill(df, time="year", series=["a", "b"])
    assert res.table["year"].tolist() == [2021, 2022, 2023]
    leader = res.table["leader"]
    assert leader[0] == "b" and pd.isna(leader[1]) and leader[2] == "a"


def test_animated_bubble_nullable_columns_and_missing_colour():
    df = pd.DataFrame({"t": [1, 1, 1], "x": [0.0, 1.0, np.nan], "y": [0.0, 1.0, 2.0], "s": [1.0, 2.0, 3.0],
                       "c": ["a", None, "a"], "lab": ["p", "q", "r"]}).convert_dtypes()
    res = vc.animated_bubble(df, time="t", x="x", y="y", size="s", color="c", label="lab")
    res.figure.canvas.draw()
    assert "(missing)" in [t.get_text() for t in res.axes.get_legend().get_texts()]


def test_gantt_datetime_groups_are_coloured_and_missing_group_in_legend():
    proj = datasets.projects()
    proj["phase"] = pd.to_datetime(["2024-01-01", "2024-02-01"] * 3)
    res = vc.gantt(proj, task="task", start="start", end="end", group="phase")
    colours = {tuple(np.round(p.get_facecolor(), 3)) for p in res.axes.patches}
    assert len(colours) == 2
    proj.loc[0, "team"] = np.nan
    res = vc.gantt(proj, task="task", start="start", end="end", group="team")
    assert "(missing)" in [t.get_text() for t in res.axes.get_legend().get_texts()]


def test_pca_plot_keeps_categorical_group_order():
    trial = datasets.trial()
    res = vc.pca_plot(trial, features=["baseline", "score"], group="arm")
    assert [t.get_text() for t in res.axes.get_legend().get_texts()] == ["control", "low dose", "high dose"]
    assert isinstance(res.table["arm"].dtype, pd.CategoricalDtype)


def test_radar_nullable_metrics_and_markers():
    m = datasets.measurements()[["group", "length", "width", "depth"]].convert_dtypes()
    m.loc[m.group == "alpha", "width"] = pd.NA
    res = vc.radar(m, metrics=["length", "width", "depth"], group="group")
    assert np.isnan(res.table.set_index("group").loc["alpha", "width_scaled"])
    assert all(line.get_marker() == "o" for line in res.axes.get_lines())


def test_quadrant_plot_rejects_one_sided_median():
    df = pd.DataFrame({"x": [0] * 8 + [5, 9], "y": np.arange(10.0)})
    with pytest.raises(ValueError, match="one side"):
        vc.quadrant_plot(df, x="x", y="y", center="median")


def test_network_categorical_columns_and_labels():
    base = pd.DataFrame({"s": ["A", "B", "C"], "t": ["B", "C", "D"], "w": [1.0, 2.0, 3.0]})
    cat = base.assign(s=base.s.astype("category"), t=base.t.astype("category"))
    for directed in (False, True):
        g = vc.network_map(cat, "s", "t", weight="w", directed=directed).info["graph"]
        assert g.number_of_edges() == 3
    headerless = pd.DataFrame([("A", "B", 3), ("A", "C", 2), ("B", "C", 1)])
    g = vc.network_map(headerless, 0, 1, weight=2).info["graph"]
    assert g["A"]["B"][2] == 3
    with pytest.raises(ValueError, match="missing"):
        vc.network_map(base.assign(w=[1.0, np.nan, 2.0]), "s", "t", weight="w")


def test_network_keeps_node_order_and_sums_reciprocal_weights_for_communities():
    e = pd.DataFrame({"s": ["Zoe", "Amy", "Mia"], "t": ["Amy", "Mia", "Bob"]})
    assert vc.network_map(e, "s", "t").table["node"].tolist() == ["Zoe", "Amy", "Mia", "Bob"]
    d = pd.DataFrame({"s": ["A", "B", "C", "D", "B", "D"], "t": ["B", "A", "D", "C", "C", "A"], "w": [50, 1, 50, 1, 10, 10]})
    comm = vc.network_map(d, "s", "t", weight="w", directed=True).table.set_index("node")["community"]
    assert comm["A"] == comm["B"] and comm["C"] == comm["D"] and comm["A"] != comm["C"]


def test_dumbbell_keeps_integer_columns():
    df = pd.DataFrame({"k": ["a"], "s": [2**53 + 1], "e": [2**53 + 3]})
    t = vc.dumbbell(df, label="k", start="s", end="e").table
    assert t["change"].iloc[0] == 2 and t["s"].dtype == np.int64


def test_bullet_negative_values_and_array_bands():
    df = pd.DataFrame({"k": ["profit"], "v": [-5.0], "t": [10.0]})
    res = vc.bullet(df, label="k", value="v", target="t", bands=np.array([0.0, 5.0, 15.0]))
    assert res.axes[0].get_xlim()[0] < -5


def test_waterfall_uses_start_label():
    df = pd.DataFrame({"m": ["Jan", "Feb", "Mar"], "v": [100.0, 20.0, -5.0]})
    t = vc.waterfall(df, label="m", value="v", start_label="Opening balance").table
    assert t["label"].tolist() == ["Opening balance", "Feb", "Mar", "Total"]


def test_donut_grid_skips_boolean_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "flag": [True, False]})
    assert set(vc.donut_grid(df).table["category"]) == {"a", "b"}


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_upset_max_intersections_validated(bad):
    with pytest.raises(ValueError, match="max_intersections"):
        vc.upset({"A": {1, 2}, "B": {2, 3}}, max_intersections=bad)


# --- third verification round ------------------------------------------------------------------------------------


def _ticks(ax, axis="y"):
    ax.figure.canvas.draw()
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    k = 1 if axis == "y" else 0
    return {round(t.get_position()[k], 6): t.get_text() for t in labels if lo <= t.get_position()[k] <= hi}


def test_abbrev_formatter_shows_true_values_at_2_5_steps():
    res = vc.waterfall(pd.DataFrame({"s": ["Start", "Q1", "Q2", "Q3", "Q4"], "d": [12000, 3000, -2500, 4000, 1200]}),
                       label="s", value="d")
    ticks = _ticks(res.axes)
    assert ticks[2500] == "2.5K" and ticks[7500] == "7.5K" and ticks[17500] == "17.5K"


def test_abbrev_formatter_log_axis_and_narrow_range():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    d = pd.DataFrame({"f": list("abcd"), "a": [150, 2_000, 40_000, 900_000], "b": [300, 3_500, 60_000, 2_400_000]})
    vc.dumbbell(d, label="f", start="a", end="b", ax=ax)
    ticks = _ticks(ax, "x")
    assert ticks[100] == "100" and ticks[1_000_000] == "1M"
    d = pd.DataFrame({"c": list("abc"), "a": [1_200_150, 1_200_400, 1_200_700], "b": [1_200_300, 1_200_650, 1_200_900]})
    res = vc.dumbbell(d, label="c", start="a", end="b")
    labels = list(_ticks(res.axes, "x").values())
    assert len(set(labels)) == len(labels)
    assert res.axes.xaxis.get_offset_text().get_text() != ""  # Matplotlib's offset carries the magnitude


def test_fd_keeps_outliers_and_heavy_tails():
    rng = np.random.default_rng(0)
    income = np.r_[rng.normal(50_000, 15_000, 999), 25_000_000]
    assert st._bin_edges(income, "fd")[1] == "fd"
    assert st._bin_edges(rng.standard_cauchy(1000), "fd")[1] == "fd"
    assert "IQR is 0" in vc.histogram(pd.DataFrame({"x": np.r_[np.zeros(80), np.arange(1, 21)]}), x="x").info["bin_rule"]


def test_save_to_file_like_objects():
    from io import BytesIO

    buf = BytesIO()
    assert vc.waffle(datasets.survey(), category="team").save(buf) is buf and buf.getbuffer().nbytes > 0


def test_network_weight_column_labelled_zero_and_infinite_weights():
    e = pd.DataFrame([(5.0, "A", "B"), (1.0, "B", "C"), (1.0, "A", "C")])
    res = vc.network_map(e, 1, 2, weight=0)
    assert res.info["graph"]["A"]["B"][0] == 5.0
    assert res.table.set_index("node").loc["A", "strength"] == 6.0
    bad = e.copy()
    bad[0] = [np.inf, 1.0, 1.0]
    with pytest.raises(ValueError, match="finite"):
        vc.network_map(bad, 1, 2, weight=0)


def test_weighted_betweenness_splits_equal_paths_exactly():
    # 1/2 + 1/12 == 1/3 + 1/4: both brokers carry half of the S-E shortest paths
    e = pd.DataFrame({"s": ["S", "B", "S", "C"], "t": ["B", "E", "C", "E"], "w": [2, 12, 3, 4]})
    bc = vc.network_map(e, "s", "t", weight="w").table.set_index("node")["betweenness"]
    assert bc["B"] == pytest.approx(bc["C"]) and bc["B"] > 0


def test_directed_network_draws_arrows():
    e = pd.DataFrame({"s": ["A", "B"], "t": ["B", "C"]})
    res = vc.network_map(e, "s", "t", directed=True)
    assert any(p.__class__.__name__ == "FancyArrowPatch" for p in res.axes.patches)
    pytest.importorskip("plotly")
    fig = vc.network_map(e, "s", "t", directed=True, interactive=True).figure
    assert len(fig.layout.annotations) == 2 and all(a.showarrow for a in fig.layout.annotations)


def test_day_first_dates_raise_instead_of_being_guessed():
    df = pd.DataFrame({"when": ["01/02/2024", "15/02/2024", "28/02/2024"], "v": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="consistent format"):
        vc.period_bars(df, date="when", value="v")


def test_timeseries_fill_keeps_distinct_instants_across_dst():
    t = pd.date_range("2024-11-03 00:00", periods=4, freq="h", tz="America/New_York")  # 01:00 occurs twice
    df = pd.DataFrame({"t": t, "a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 2.0, 2.0, 2.0]})
    res = vc.timeseries_fill(df, time="t", series=["a", "b"])
    assert len(res.table) == 4 and res.table["t"].dt.tz is not None


def test_timeline_durations_use_real_elapsed_time():
    tz = "America/New_York"
    g = pd.DataFrame({"task": ["A"], "s": [pd.Timestamp("2024-03-09 12:00", tz=tz)],
                      "e": [pd.Timestamp("2024-03-10 12:00", tz=tz)]})
    assert vc.gantt(g, "task", "s", "e").table["duration_days"].iloc[0] == pytest.approx(23 / 24)
    x = pd.DataFrame({"task": ["deploy"], "s": pd.to_datetime(["2024-06-01 22:00"]).tz_localize("UTC"),
                      "e": pd.to_datetime(["2024-06-01 20:00"]).tz_localize(tz)})
    assert vc.gantt(x, "task", "s", "e").table["duration_days"].iloc[0] == pytest.approx(2 / 24)
    with pytest.raises(ValueError, match="mix"):
        vc.gantt(g.assign(e=pd.Timestamp("2024-03-10 12:00")), "task", "s", "e")


@pytest.mark.parametrize("years", [pd.Categorical([2020, 2021, 2022]), pd.Series([2020, "2021", 2022], dtype=object)])
def test_timeseries_fill_numeric_years_stay_numeric(years):
    df = pd.DataFrame({"year": years, "a": [1.0, 3.0, 2.0], "b": [2.0, 1.0, 3.0]})
    assert vc.timeseries_fill(df, time="year", series=["a", "b"]).table["year"].tolist() == [2020, 2021, 2022]


def test_animated_bubble_labels_only_drawn_points_and_rejects_all_missing_size():
    df = pd.DataFrame({"t": [1, 1, 1], "x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0], "s": [1.0, np.nan, 2.0],
                       "lab": ["p", "q", None]})
    res = vc.animated_bubble(df, time="t", x="x", y="y", size="s", label="lab")
    assert [t.get_text() for t in res.axes.texts if t.get_text() in ("p", "q", "None", "nan")] == ["p"]
    with pytest.raises(ValueError, match="no non-missing"):
        vc.animated_bubble(df.assign(s=np.nan), time="t", x="x", y="y", size="s")


@pytest.mark.parametrize("dtype,start,end,change", [("uint8", 5, 3, -2), ("int8", 100, -100, -200)])
def test_dumbbell_small_and_unsigned_integers_do_not_wrap(dtype, start, end, change):
    df = pd.DataFrame({"k": ["a"], "s": np.array([start], dtype=dtype), "e": np.array([end], dtype=dtype)})
    assert vc.dumbbell(df, label="k", start="s", end="e").table["change"].iloc[0] == change


def test_dumbbell_nullable_with_missing():
    df = pd.DataFrame({"k": ["a", "b"], "s": pd.array([1, None], dtype="Int64"), "e": pd.array([2, 3], dtype="Int64")})
    t = vc.dumbbell(df, label="k", start="s", end="e", sort=False).table
    assert t["change"].iloc[0] == 1 and np.isnan(t["change"].iloc[1])


def test_quadrant_points_on_the_mean_count_as_high_with_standardization():
    x = [5, 3, 1, 5, 2, 2, 4, 4, 3, 1, 3, 3, 4, 2, 3, 2, 1, 4, 5, 2, 1, 3, 2, 2, 4, 5, 1, 2, 3, 2, 4, 4, 3, 2, 2, 4, 5, 5, 5, 2]
    df = pd.DataFrame({"x": x, "y": np.arange(40) % 7})
    a = vc.quadrant_plot(df, "x", "y").table["n"].tolist()
    b = vc.quadrant_plot(df, "x", "y", standardize=False).table["n"].tolist()
    assert a == b


def test_quadrant_error_leaves_no_half_drawn_figure():
    import matplotlib.pyplot as plt

    before = len(plt.get_fignums())
    with pytest.raises(ValueError):
        vc.quadrant_plot(pd.DataFrame({"x": [0] * 8 + [5, 9], "y": np.arange(10.0)}), "x", "y", center="median")
    assert len(plt.get_fignums()) == before


def test_bullet_all_negative_keeps_zero_and_target_visible():
    df = pd.DataFrame({"k": ["loss"], "v": [-8.0], "t": [-5.0]})
    lo, hi = vc.bullet(df, label="k", value="v", target="t", bands=[-10.0, -6.0]).axes[0].get_xlim()
    assert lo < -8 and hi >= 0


# --- fourth verification round -----------------------------------------------------------------------------------


def test_abbrev_formatter_float_noise_at_huge_magnitudes_uses_offset():
    sent = np.array([1727500000.120, 1727500000.310, 1727500000.550])
    d = pd.DataFrame({"r": ["a", "b", "c"], "s": sent, "e": sent + [0.045, 0.080, 0.130]})
    res = vc.dumbbell(d, label="r", start="s", end="e")
    labels = list(_ticks(res.axes, "x").values())
    assert len(set(labels)) == len(labels)


def test_abbrev_formatter_absolute_never_negative():
    d = pd.DataFrame({"age": ["0-9", "10-19"], "m": [1_200_150, 1_200_700], "f": [1_200_300, 1_200_900]})
    res = vc.divergent_bar(d, category="age", left="m", right="f")
    res.axes.set_xlim(-1_200_800, -1_200_100)
    labels = list(_ticks(res.axes, "x").values())
    offset = res.axes.xaxis.get_offset_text().get_text()
    assert not any(t.startswith(("-", "−")) for t in labels + [offset])
    scale = float(offset.replace("\u00d7", "")) if offset else 1.0  # Matplotlib may show a "1e6" multiplier
    assert all(1_200_000 <= float(t.replace(",", "")) * scale <= 1_200_900 for t in labels if t)


def test_fd_reason_is_accurate():
    x = np.r_[np.linspace(0, 1, 999), 1e9]  # healthy IQR, one astronomical outlier
    edges, rule = st._bin_edges(x, "fd")
    assert rule.startswith("sturges (FD would need")


def test_timeseries_fill_categorical_and_nullable_time_columns():
    df = pd.DataFrame({"year": pd.Categorical([2020, 2021, None, 2022]), "a": [1.0, 3.0, 5.0, 2.0], "b": [2.0] * 4})
    assert vc.timeseries_fill(df, time="year", series=["a", "b"]).table["year"].tolist() == [2020, 2021, 2022]
    df = pd.DataFrame({"t": pd.array([1, 2, 3], dtype="Int64"), "a": [1.0, 3.0, 2.0], "b": [2.0, 2.0, 2.0]})
    assert vc.timeseries_fill(df, time="t", series=["a", "b"]).info["crossovers"] == 1


def test_keep_tz_rejects_offset_and_offsetless_mixture():
    g = pd.DataFrame({"task": ["A", "B"], "s": ["2024-06-03 09:00-04:00", "2024-06-04 09:00-04:00"],
                      "e": ["2024-06-03 17:00-04:00", "2024-06-04 17:00"]})
    with pytest.raises(ValueError, match="with and without a UTC offset"):
        vc.gantt(g, "task", "s", "e")


def test_integer_columns_are_not_read_as_dates():
    df = pd.DataFrame({"d": [20240101, 20240102], "v": [1.0, 2.0]})
    with pytest.raises(ValueError, match="numbers"):
        vc.period_bars(df, date="d", value="v")


def test_gantt_today_on_a_dst_change_day():
    g = pd.DataFrame({"task": ["A"], "s": [pd.Timestamp("2024-10-01", tz="America/Asuncion")],
                      "e": [pd.Timestamp("2024-10-10", tz="America/Asuncion")]})
    vc.gantt(g, "task", "s", "e", today="2024-10-06")  # midnight does not exist that day


def test_duration_plot_with_an_empty_inner_column():
    p = datasets.projects()
    for c in ("start", "end", "work_end"):
        p[c] = p[c].dt.tz_localize("UTC")
    p["work_start"] = pd.NaT  # an all-empty (naive) column must not count as a zone mismatch
    t = vc.duration_plot(p, label="task", start="start", end="end", inner_start="work_start", inner_end="work_end").table
    assert t["inner_days"].isna().iloc[0] and t["outer_days"].notna().all()


def test_betweenness_ties_with_decimal_weights():
    for w in ([0.3, 0.6, 0.2], [0.15, 0.3, 0.1]):
        e = pd.DataFrame({"s": ["A", "B", "A"], "t": ["B", "C", "C"], "w": w})
        bc = vc.network_map(e, "s", "t", weight="w").table.set_index("node")["betweenness"]
        assert bc["B"] == pytest.approx(0.5)


def test_calendar_heatmap_value_column_labelled_zero():
    d = pd.DataFrame({"when": pd.date_range("2024-01-01", periods=5), 0: [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert len(vc.calendar_heatmap(d, date="when", value=0).table) == 5


def test_plotly_save_format_and_file_like(tmp_path):
    pytest.importorskip("plotly")
    from io import StringIO

    res = vc.sankey(pd.DataFrame({"a": ["x"], "b": ["y"], "v": [1]}), source="a", target="b", value="v")
    assert res.save(str(tmp_path / "flow"), format="html").endswith("flow.html")
    buf = StringIO()
    res.save(buf)
    assert "<html" in buf.getvalue().lower()


def test_directed_reciprocal_edges_are_both_visible():
    e = pd.DataFrame({"s": ["A", "B"], "t": ["B", "A"], "w": [4.0, 1.0]})
    res = vc.network_map(e, "s", "t", weight="w", directed=True)
    arrows = [p for p in res.axes.patches if p.__class__.__name__ == "FancyArrowPatch"]
    assert len(arrows) == 2
    paths = {tuple(np.round(a.get_path().vertices.mean(axis=0), 3)) for a in arrows}
    assert len(paths) == 2  # the two arrows follow different curves


def test_bullet_bands_start_at_the_axis_start():
    df = pd.DataFrame({"k": ["loss"], "v": [-8.0], "t": [-5.0]})
    ax = vc.bullet(df, label="k", value="v", target="t", bands=[-10.0, -6.0]).axes[0]
    band_starts = [patch.get_x() for patch in ax.patches[:2]]
    assert band_starts == [pytest.approx(-10.0), pytest.approx(-10.0)]  # the (-10, -6] band starts at -10, not 0


def test_quadrant_point_exactly_on_decimal_mean_is_high():
    df = pd.DataFrame({"x": [0.1, 0.2, 0.3], "y": [1.2, 2.2, 3.2]})
    shares = dict(zip(*vc.quadrant_plot(df, "x", "y").table[["quadrant", "n"]].T.values))
    assert shares["high x, high y"] == 2 and shares["low x, low y"] == 1


def test_legend_title_for_column_labelled_zero():
    df = pd.DataFrame({0: ["a", "b"] * 10, "v": np.arange(20.0)})
    res = vc.histogram(df, x="v", hue=0)
    assert res.axes.flat[0].get_legend().get_title().get_text() == "0"


def test_dumbbell_nullable_int64_stays_exact():
    big = 2**53
    df = pd.DataFrame({"k": ["a"], "s": pd.array([big + 1], dtype="Int64"), "e": pd.array([big + 3], dtype="Int64")})
    assert vc.dumbbell(df, label="k", start="s", end="e").table["change"].iloc[0] == 2


# --- fifth verification round ------------------------------------------------------------------------------------


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"s": ["A", "B", "A"], "t": ["B", "C", "C"], "w": np.array([0.3, 0.6, 0.2], dtype="float32")}),
    pd.DataFrame({"s": ["A", "A", "A", "B", "A"], "t": ["B", "B", "B", "C", "C"], "w": [0.1, 0.1, 0.1, 0.6, 0.2]}),
])
def test_betweenness_ties_survive_float32_and_summed_rows(frame):
    bc = vc.network_map(frame, "s", "t", weight="w").table.set_index("node")["betweenness"]
    assert bc["B"] == pytest.approx(0.5)


def test_plotly_save_to_binary_buffer():
    pytest.importorskip("plotly")
    from io import BytesIO

    buf = BytesIO()
    vc.sankey(pd.DataFrame({"a": ["x"], "b": ["y"], "v": [1]}), source="a", target="b", value="v").save(buf)
    assert b"<html" in buf.getvalue().lower()


def test_blank_csv_date_column_is_allowed():
    from io import StringIO

    csv = "task,start,end,ws,we\nA,2024-01-01,2024-01-05,,\nB,2024-01-03,2024-01-09,,\n"
    p = pd.read_csv(StringIO(csv))  # ws/we are float NaN columns
    t = vc.duration_plot(p, label="task", start="start", end="end", inner_start="ws", inner_end="we").table
    assert t["outer_days"].tolist() == [4.0, 6.0] and t["inner_days"].isna().all()


@pytest.mark.parametrize("col", [pd.Series([20240101, 20240102], dtype=object), pd.Categorical([2020, 2021])])
def test_numbers_in_object_or_categorical_columns_are_not_dates(col):
    with pytest.raises(ValueError, match="numbers"):
        vc.period_bars(pd.DataFrame({"d": col, "v": [1.0, 2.0]}), date="d", value="v")


def test_mixed_offsets_with_blanks():
    g = pd.DataFrame({"task": ["A", "B", "C"],
                      "s": ["2024-03-09 09:00-05:00", "2024-03-11 09:00-04:00", "2024-03-12 09:00-04:00"],
                      "e": ["2024-03-09 17:00-05:00", "", "2024-03-12 17:00-04:00"]})
    t = vc.gantt(g.iloc[[0, 2]], "task", "s", "e").table
    assert t["duration_days"].tolist() == [pytest.approx(8 / 24)] * 2
    from viz_calc.timeseries import _dates

    assert _dates(g["e"], "e", keep_tz=True).isna().tolist() == [False, True, False]


def test_empty_tz_aware_column_next_to_naive_ones():
    p = datasets.projects()
    p["work_start"] = pd.Series(pd.NaT, index=p.index, dtype="datetime64[ns, UTC]")
    t = vc.duration_plot(p, label="task", start="start", end="end", inner_start="work_start", inner_end="work_end").table
    assert t["inner_days"].isna().all()


@pytest.mark.parametrize("values", [[10.0] * 5, [1.0] * 5])
def test_centered_bar_all_above_or_all_below(values):
    df = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 5, "v": values + [0.0] * 5})
    t = vc.centered_bar(df, x="g", y="v", threshold=5).table
    assert t["ci_low"].min() >= 0 and t["ci_high"].max() <= 1


def test_wilson_exact_at_the_boundaries():
    assert st.wilson_ci(0, 7) [0] == 0.0 and st.wilson_ci(7, 7)[1] == 1.0


# --- sixth verification round ------------------------------------------------------------------------------------


def test_plotly_save_to_text_mode_tempfile(tmp_path):
    pytest.importorskip("plotly")
    import tempfile

    res = vc.sankey(pd.DataFrame({"a": ["x"], "b": ["y"], "v": [1]}), source="a", target="b", value="v")
    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=tmp_path, delete=False, encoding="utf-8") as f:
        res.save(f)
    assert "<html" in open(f.name, encoding="utf-8").read().lower()


def test_rational_weights_summed_from_rows_keep_ties():
    # three rows of 1/3 make a weight of exactly 1, tying A-B (distance 1) with A-C-B (1/2 + 1/2)
    e = pd.DataFrame({"s": ["A", "A", "A", "A", "C"], "t": ["B", "B", "B", "C", "B"],
                      "w": [1 / 3, 1 / 3, 1 / 3, 2.0, 2.0]})
    bc = vc.network_map(e, "s", "t", weight="w").table.set_index("node")["betweenness"]
    assert bc["C"] == pytest.approx(0.5)


def test_nullable_float32_weights_keep_ties():
    e = pd.DataFrame({"s": ["A", "B", "A"], "t": ["B", "C", "C"], "w": pd.array([0.3, 0.6, 0.2], dtype="Float32")})
    bc = vc.network_map(e, "s", "t", weight="w").table.set_index("node")["betweenness"]
    assert bc["B"] == pytest.approx(0.5)


def test_date_column_with_a_stray_number_raises():
    import datetime as dt

    df = pd.DataFrame({"date": [dt.datetime(2024, 1, 5), dt.datetime(2024, 2, 9), 45332], "v": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="numbers"):
        vc.period_bars(df, date="date", value="v")
