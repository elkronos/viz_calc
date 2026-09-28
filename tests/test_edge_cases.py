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
