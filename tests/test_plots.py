"""Behaviour of the plotting functions, including regressions for bugs in the original scripts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets


@pytest.fixture(scope="module")
def trial():
    return datasets.trial()


@pytest.fixture(scope="module")
def meas():
    return datasets.measurements()


@pytest.fixture(scope="module")
def survey():
    return datasets.survey()


LEVELS = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]


def _calls():
    """Every public plotting function with valid arguments (used for the generic contract tests)."""
    trial, meas, survey = datasets.trial(), datasets.measurements(), datasets.survey()
    sales, proj, edges = datasets.sales(), datasets.projects(), datasets.network()
    small = pd.DataFrame({"k": list("abcd"), "a": [3.0, 5, 2, 8], "b": [4.0, 1, 6, 7]})
    return {
        "estimation_plot": (vc.estimation_plot, trial, dict(x="arm", y="score", n_resamples=500)),
        "benchmark_bar": (vc.benchmark_bar, trial, dict(x="arm", y="score", threshold=50)),
        "lollipop": (vc.lollipop, survey, dict(x="team", y="tenure_years")),
        "dumbbell": (vc.dumbbell, small, dict(label="k", start="a", end="b")),
        "divergent_bar": (vc.divergent_bar, small, dict(category="k", left="a", right="b")),
        "centered_bar": (vc.centered_bar, trial, dict(x="arm", y="score")),
        "likert": (vc.likert, survey, dict(items=["Tools are adequate", "Meetings are useful"], levels=LEVELS)),
        "correlogram": (vc.correlogram, meas, dict()),
        "compare_correlations": (vc.compare_correlations, meas, dict(group="group")),
        "quadrant_plot": (vc.quadrant_plot, trial, dict(x="baseline", y="score")),
        "histogram": (vc.histogram, trial, dict(x="score")),
        "ridgeplot": (vc.ridgeplot, meas, dict(x="depth", group="group")),
        "raincloud": (vc.raincloud, trial, dict(x="arm", y="score")),
        "waffle": (vc.waffle, survey, dict(category="team")),
        "percent_grid": (vc.percent_grid, trial, dict(column="improved")),
        "stacked_percentages": (vc.stacked_percentages, survey, dict(group="team", category="remote")),
        "donut_grid": (vc.donut_grid, small.set_index("k"), dict()),
        "nested_pie": (vc.nested_pie, survey, dict(outer="team", inner="remote")),
        "circular_bar": (vc.circular_bar, small, dict(label="k", value="a")),
        "waterfall": (vc.waterfall, small, dict(label="k", value="a")),
        "funnel": (vc.funnel, small.sort_values("a", ascending=False), dict(stage="k", value="a")),
        "bullet": (vc.bullet, small, dict(label="k", value="a", target="b", bands=[2, 5, 9])),
        "period_bars": (vc.period_bars, sales, dict(date="date", value="online")),
        "timeseries_fill": (vc.timeseries_fill, sales.head(60), dict(time="date", series=["online", "store"])),
        "calendar_heatmap": (vc.calendar_heatmap, sales, dict(date="date", value="online")),
        "gantt": (vc.gantt, proj, dict(task="task", start="start", end="end", group="team")),
        "duration_plot": (vc.duration_plot, proj, dict(label="task", start="start", end="end",
                                                       inner_start="work_start", inner_end="work_end")),
        "pca_plot": (vc.pca_plot, meas, dict(features=["length", "width", "depth", "mass"], group="group")),
        "radar": (vc.radar, meas, dict(metrics=["length", "width", "depth", "mass"], group="group")),
        "network_map": (vc.network_map, edges, dict(source="source", target="target", weight="strength")),
        "animated_bubble": (vc.animated_bubble, sales.head(20).melt(id_vars="date", value_vars=["online", "store"])
                            .assign(size=lambda d: d["value"]), dict(time="date", x="value", y="size", size="size",
                                                                     color="variable")),
        "profile_bars": (vc.profile_bars, meas, dict()),
        "profile_boxes": (vc.profile_boxes, meas, dict()),
        "profile_scatters": (vc.profile_scatters, meas, dict()),
    }


CALLS = _calls()


@pytest.mark.parametrize("name", list(CALLS))
def test_returns_result_and_does_not_mutate_input(name):
    func, data, kwargs = CALLS[name]
    before = data.copy(deep=True)
    res = func(data, **kwargs)
    assert isinstance(res, vc.VizResult)
    assert isinstance(res.table, pd.DataFrame) and not res.table.empty
    figs = res.figure if isinstance(res.figure, list) else [res.figure]
    assert all(isinstance(f, plt.Figure) for f in figs)
    pd.testing.assert_frame_equal(data, before)  # the original scripts added columns to the caller's data


@pytest.mark.parametrize("name", [n for n in CALLS if n not in {"donut_grid", "upset"}])
def test_missing_column_raises_keyerror(name):
    func, data, kwargs = CALLS[name]
    bad = {k: ("nope" if isinstance(v, str) and v in data.columns else v) for k, v in kwargs.items()}
    if bad == kwargs:
        pytest.skip("no column argument")
    with pytest.raises(KeyError, match="nope"):
        func(data, **bad)


def test_non_dataframe_rejected():
    with pytest.raises(TypeError):
        vc.benchmark_bar([1, 2, 3], x="a", y="b")


def test_nothing_is_shown_or_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(1))
    vc.waffle(datasets.survey(), category="team")
    assert not shown and not list(tmp_path.iterdir())


def test_importing_has_no_side_effects(monkeypatch):
    import importlib

    monkeypatch.setattr(plt, "show", lambda *a, **k: (_ for _ in ()).throw(AssertionError("show called")))
    for mod in ["compare", "composition", "correlation", "distribution", "explore", "multivariate", "network", "timeseries"]:
        importlib.reload(importlib.import_module(f"viz_calc.{mod}"))


# --- statistics shown on the plots ------------------------------------------------------------------------------


def test_estimation_plot_comparisons(trial):
    res = vc.estimation_plot(trial, x="arm", y="score", n_resamples=1000)
    comps = res.info["comparisons"]
    assert list(comps["group"]) == ["low dose", "high dose"]
    assert (comps["ci_low"] < comps["difference"]).all() and (comps["difference"] < comps["ci_high"]).all()
    assert (comps["p_adjusted"] >= comps["p"]).all()
    ctrl = trial.loc[trial.arm == "control", "score"]
    hi = trial.loc[trial.arm == "high dose", "score"]
    assert comps.set_index("group").loc["high dose", "difference"] == pytest.approx(hi.mean() - ctrl.mean())
    assert list(res.table["n"]) == [40, 40, 40]


def test_estimation_plot_custom_reference(trial):
    res = vc.estimation_plot(trial, x="arm", y="score", reference="high dose", n_resamples=500)
    assert set(res.info["comparisons"]["group"]) == {"control", "low dose"}
    with pytest.raises(ValueError):
        vc.estimation_plot(trial, x="arm", y="score", reference="placebo")


def test_benchmark_classifies_by_ci():
    df = pd.DataFrame({"g": ["lo"] * 30 + ["mid"] * 30 + ["hi"] * 30,
                       "v": np.r_[np.full(30, 1.0), np.linspace(4, 6, 30), np.full(30, 9.0)]})
    df.loc[0, "v"] = 1.1  # avoid zero variance
    df.loc[60, "v"] = 9.1
    res = vc.benchmark_bar(df, x="g", y="v", threshold=5)
    status = dict(zip(res.table["g"], res.table["status"]))
    assert status == {"hi": "above", "lo": "below", "mid": "indistinguishable"}


def test_lollipop_labels_without_threshold():
    # original lollipop_plot raised KeyError('label_color') when labels=True and threshold=None
    df = pd.DataFrame({"c": list("aabbc"), "v": [1, 3, 2, 2, -4]})
    res = vc.lollipop(df, x="c", y="v", labels=True)
    assert dict(zip(res.table["c"], res.table["value"])) == {"a": 2, "b": 2, "c": -4}


def test_lollipop_rejects_unknown_stat():
    # original silently fell back to the mean
    with pytest.raises(ValueError):
        vc.lollipop(pd.DataFrame({"c": ["a"], "v": [1]}), x="c", y="v", stat="mode")


def test_centered_bar_uses_wilson(trial):
    res = vc.centered_bar(trial, x="arm", y="score", threshold=55)
    row = res.table.iloc[0]
    lo, hi = vc.stats.wilson_ci(row["n_above"], row["n"])
    assert (row["ci_low"], row["ci_high"]) == pytest.approx((float(lo), float(hi)))
    assert row["p_above"] + row["p_below"] == pytest.approx(1)


def test_likert_shares_and_net(survey):
    res = vc.likert(survey, items=["Meetings are useful"], levels=LEVELS)
    row = res.table.iloc[0]
    assert row[LEVELS].sum() == pytest.approx(100)
    assert row["net"] == pytest.approx(row["Agree"] + row["Strongly agree"] - row["Disagree"] - row["Strongly disagree"])
    with pytest.raises(ValueError, match="not listed"):
        vc.likert(survey, items=["Meetings are useful"], levels=LEVELS[:4])


def test_correlogram_adjusts_over_unique_pairs(meas):
    res = vc.correlogram(meas, columns=["length", "width", "depth", "mass"], p_adjust="bonferroni")
    assert len(res.table) == 6  # k(k-1)/2 pairs, not k^2 cells
    expected = np.minimum(res.table["p"] * 6, 1)
    assert res.table["p_adjusted"].to_numpy() == pytest.approx(expected.to_numpy())
    assert res.info["matrix"].shape == (4, 4)


def test_compare_correlations_uses_fisher_z(meas):
    res = vc.compare_correlations(meas, group="group", columns=["length", "width"])
    row = res.table.iloc[0]
    z, p = vc.stats.compare_correlations_test(row["r_b"], row["n_b"], row["r_a"], row["n_a"])
    assert row["z"] == pytest.approx(float(z))
    assert row["difference"] == pytest.approx(row["r_b"] - row["r_a"])
    assert len(res.table) == 3  # 3 group pairs x 1 variable pair


def test_quadrant_percentages_are_placed_correctly():
    # original quadrant_norm printed the (low x, low y) share in the top-left corner
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8], "y": [8, 7, 6, 5, 4, 3, 2, 1]})
    res = vc.quadrant_plot(df, x="x", y="y")
    shares = dict(zip(res.table["quadrant"], res.table["percent"]))
    assert shares["low x, high y"] == 50 and shares["high x, low y"] == 50
    assert shares["high x, high y"] == 0 and shares["low x, low y"] == 0
    assert res.info["correlation"]["r"] == pytest.approx(-1)


# --- composition -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("values", [[1, 1, 1], [2, 2, 2, 2, 2, 2, 2], [98, 1, 1], [33, 33, 34]])
def test_waffle_grid_is_exactly_full(values):
    # original create_waffle crashed (IndexError) when independent rounding over-filled the grid
    df = pd.DataFrame({"c": [f"c{i}" for i in range(len(values))], "v": values})
    res = vc.waffle(df, category="c", value="v")
    assert res.table["tiles"].sum() == 100


def test_percent_grid(trial):
    res = vc.percent_grid(trial, column="improved", facet="arm")
    assert list(res.table["facet"]) == ["control", "low dose", "high dose"]
    assert (res.table["ci_low"] <= res.table["percent"]).all()
    with pytest.raises(ValueError, match="success"):
        vc.percent_grid(pd.DataFrame({"a": ["yes", "no"]}), column="a")
    res2 = vc.percent_grid(pd.DataFrame({"a": ["yes", "no", "yes", "yes"]}), column="a", success="yes")
    assert res2.table["percent"].iloc[0] == 75


def test_stacked_percentages_sum_to_100(survey):
    res = vc.stacked_percentages(survey, group="team", category="remote")
    assert res.table.groupby("team")["percent"].sum().to_numpy() == pytest.approx(100)


def test_waterfall_levels_and_changes():
    df = pd.DataFrame({"k": ["start", "a", "b", "end"], "v": [100, 120, 90, 150]})
    res = vc.waterfall(df, label="k", value="v", values_are="levels")
    t = res.table
    assert list(t["change"]) == [100, 20, -30, 60, 150]
    assert t["end"].iloc[-1] == 150
    res2 = vc.waterfall(df.assign(v=[100, 20, -30, 60]), label="k", value="v", start_label="start", total_label=None)
    assert res2.table["end"].iloc[-1] == 150


def test_funnel_rates():
    df = pd.DataFrame({"s": ["a", "b", "c"], "n": [200, 100, 25]})
    t = vc.funnel(df, stage="s", value="n").table
    assert list(t["pct_of_first"]) == [100, 50, 12.5]
    assert list(t["pct_of_previous"]) == [100, 50, 25]
    assert list(t["drop_off"]) == [0, 100, 75]


def test_upset_intersections_partition_the_union():
    sets = {"A": {1, 2, 3, 4}, "B": {3, 4, 5}, "C": {4, 6}}
    res = vc.upset(sets)
    assert res.table["size"].sum() == 6
    row = res.table[(res.table.A) & (res.table.B) & (res.table.C)]
    assert row["size"].item() == 1
    assert res.info["set_sizes"].to_dict() == {"A": 4, "B": 3, "C": 2}


def test_nested_pie_percentages(survey):
    t = vc.nested_pie(survey, outer="team", inner="remote").table
    assert t["percent_of_total"].sum() == pytest.approx(100)
    assert t.groupby("team")["percent_of_parent"].sum().to_numpy() == pytest.approx(100)


def test_donut_grid_rejects_negative():
    with pytest.raises(ValueError):
        vc.donut_grid(pd.DataFrame({"a": [1, -1], "b": [2, 3]}))


# --- distributions and multivariate ------------------------------------------------------------------------------


def test_histogram_bins_shared_across_groups(trial):
    res = vc.histogram(trial, x="score", hue="arm", facet="site", ref_line="median")
    assert len(res.table) == 6
    edges = res.info["bin_edges"]
    assert edges[0] <= trial["score"].min() and edges[-1] >= trial["score"].max()


def test_pca_matches_numpy_and_explains_all_variance(meas):
    feats = ["length", "width", "depth", "mass"]
    res = vc.pca(meas, feats)
    assert res["explained_variance_ratio"].sum() == pytest.approx(1)
    corr_eig = np.sort(np.linalg.eigvalsh(meas[feats].corr().to_numpy()))[::-1]
    assert res["eigenvalues"].to_numpy() == pytest.approx(corr_eig)


def test_pca_data_ellipse_covers_expected_share():
    # original plot_pca took the square root of the eigenvalues twice, drawing ellipses far too small
    rng = np.random.default_rng(0)
    pts = rng.multivariate_normal([0, 0], [[4, 1.5], [1.5, 1]], 4000)
    df = pd.DataFrame({"a": pts[:, 0], "b": pts[:, 1], "c": rng.normal(0, 0.01, 4000)})
    res = vc.pca_plot(df, features=["a", "b", "c"], scale=False, ellipse="data", level=0.95)
    e = res.info["ellipses"][None]
    s = res.table[["PC1", "PC2"]].to_numpy()
    ang = np.deg2rad(e["angle"])
    rot = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
    u = (s - [e["cx"], e["cy"]]) @ rot.T
    inside = (u[:, 0] / (e["width"] / 2)) ** 2 + (u[:, 1] / (e["height"] / 2)) ** 2 <= 1
    assert inside.mean() == pytest.approx(0.95, abs=0.015)


def test_pca_components_validated(meas):
    with pytest.raises(ValueError):
        vc.pca_plot(meas, features=["length", "width"], components=(1, 3))


def test_radar_scaling(meas):
    t = vc.radar(meas, metrics=["length", "width", "depth"], group="group", normalize="groups").table
    assert t["length_scaled"].min() == 0 and t["length_scaled"].max() == 1
    t2 = vc.radar(meas, metrics=["length", "width", "depth"], group="group").table
    assert (t2.filter(like="_scaled") > 0).all().all() and (t2.filter(like="_scaled") < 1).all().all()


# --- time --------------------------------------------------------------------------------------------------------


def test_period_bars_keeps_empty_periods():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-04-02"]), "v": [1, 2, 5]})
    t = vc.period_bars(df, date="d", value="v", freq="month").table
    assert len(t) == 4
    assert list(t["value"]) == [3, 0, 0, 5]
    assert list(t["n"]) == [2, 0, 0, 1]


def test_timeseries_fill_counts_crossovers():
    x = np.linspace(0, 4 * np.pi, 200)
    df = pd.DataFrame({"t": x, "a": np.sin(x), "b": np.full_like(x, 0.5)})  # sin crosses 0.5 at π/6, 5π/6, 13π/6, 17π/6
    assert vc.timeseries_fill(df, time="t", series=["a", "b"]).info["crossovers"] == 4
    with pytest.raises(ValueError):
        vc.timeseries_fill(df, time="t", series=["a"])


def test_gantt_rejects_reversed_dates():
    df = pd.DataFrame({"t": ["x"], "s": ["2024-02-01"], "e": ["2024-01-01"]})
    with pytest.raises(ValueError, match="before start"):
        vc.gantt(df, task="t", start="s", end="e")


def test_animated_bubble_uses_one_area_scale():
    df = pd.DataFrame({"t": [1, 1, 2, 2], "x": [0, 1, 0, 1], "y": [0, 1, 1, 0], "s": [10, 20, 40, 5]})
    res = vc.animated_bubble(df, time="t", x="x", y="y", size="s", max_area=400)
    assert res.info["area_scale"] == pytest.approx(10)
    assert res.info["frames"] == [1, 2]


# --- network, explore, export ------------------------------------------------------------------------------------


def test_network_metrics():
    edges = datasets.network()
    res = vc.network_map(edges, source="source", target="target", weight="strength")
    t = res.table.set_index("node")
    assert res.info["n_communities"] == 2
    assert t.loc["Ana", "community"] != t.loc["Gus", "community"]
    assert t["betweenness"].idxmax() in {"Cai", "Hal", "Fay", "Gus"}


def test_network_and_sankey_interactive():
    pytest.importorskip("plotly")
    edges = datasets.network()
    res = vc.network_map(edges, source="source", target="target", interactive=True)
    assert res.axes is None and hasattr(res.figure, "to_html")
    flows = pd.DataFrame({"a": ["x", "x", "y"], "b": ["y", "z", "z"], "v": [5, 3, 2]})
    s = vc.sankey(flows, source="a", target="b", value="v")
    assert s.table.set_index("node").loc["y", "inflow"] == 5
    assert s.table.set_index("node").loc["z", "inflow"] == 5


def test_profile_boxes_pairs_categorical_with_numeric(meas):
    t = vc.profile_boxes(meas).table
    assert set(t["categorical"]) == {"group", "grade"}
    assert set(t["numeric"]) == {"length", "width", "depth", "mass"}
    assert t["eta_squared"].is_monotonic_decreasing


def test_profile_scatters_sorted_by_strength(meas):
    t = vc.profile_scatters(meas).table
    assert t["r"].abs().is_monotonic_decreasing


def test_save_and_pptx(tmp_path, meas):
    pytest.importorskip("pptx")
    res = vc.profile_bars(meas, per_page=2)
    paths = res.save(str(tmp_path / "bars.png"))
    assert len(paths) == len(res.figure) and all((tmp_path / p.split("/")[-1]).exists() for p in paths)
    out = vc.to_pptx([res, vc.waffle(meas, category="grade")], str(tmp_path / "deck.pptx"), titles=["Counts"])
    assert (tmp_path / "deck.pptx").stat().st_size > 0 and out.endswith("deck.pptx")


def test_palette():
    assert vc.palette(3) == list(vc.OKABE_ITO[:3])
    assert len(vc.palette(12)) == 12
    assert vc.palette(3, ["red"]) == ["red", "red", "red"]
