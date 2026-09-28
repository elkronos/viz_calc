"""Group comparisons and shared helpers: repeated columns, level order, missing groups and number labels."""

import math
import random
from datetime import datetime, timedelta, timezone
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets
from viz_calc._core import abbreviate, check_distinct, exact_mean, level_label


@pytest.fixture
def small():
    return pd.DataFrame({"k": list("abcd"), "a": [3.0, 5, 2, 8], "b": [4.0, 1, 6, 7]})


# --- one column passed for two roles ---------------------------------------------------------------------------


def test_check_distinct_names_both_roles_and_ignores_none():
    check_distinct(x="a", y="b", label=None, other=None)
    with pytest.raises(ValueError, match="x and label must be different columns; both are 'a'"):
        check_distinct(x="a", y="b", label="a")


@pytest.mark.parametrize("call, message", [
    (lambda d: vc.dumbbell(d, label="k", start="a", end="a"), "start and end"),
    (lambda d: vc.quadrant_plot(d, x="a", y="a"), "x and y"),
    (lambda d: vc.divergent_bar(d, category="k", left="a", right="a"), "left and right"),
    (lambda d: vc.divergent_bar(d, category="a", left="a", right="b"), "category and left"),
])
def test_one_column_for_two_value_roles_raises_a_clear_error(small, call, message):
    with pytest.raises(ValueError, match=f"{message} must be different columns"):
        call(small)


def test_a_label_column_may_also_be_a_value_column(small):
    r = vc.dumbbell(small, label="a", start="a", end="b", sort=False)
    assert r.table.columns.tolist() == ["a", "b", "change", "pct_change"]
    assert r.table["change"].tolist() == [1.0, -4.0, 4.0, -1.0]
    assert [t.get_text() for t in r.axes.get_yticklabels()] == ["3.0", "5.0", "2.0", "8.0"]
    r = vc.quadrant_plot(small, x="a", y="b", label="a")
    assert r.table["n"].sum() == 4
    assert sorted(t.get_text() for t in r.axes.texts if "\n" not in t.get_text()) == ["2.0", "3.0", "5.0", "8.0"]


def test_compare_correlations_rejects_the_group_column_in_columns():
    m = datasets.measurements()
    with pytest.raises(ValueError, match="group column 'group' cannot also be one of columns"):
        vc.compare_correlations(m, group="group", columns=["group", "length", "width"])
    r = vc.compare_correlations(m, group="group", columns=(c for c in ["length", "width"]))
    assert set(r.table["var1"]) | set(r.table["var2"]) == {"length", "width"}


# --- level order -----------------------------------------------------------------------------------------------


def _ticks(res):
    return [t.get_text() for t in res.figure[0].axes[0].get_xticklabels()]


def test_profile_boxes_orders_numeric_levels_by_value():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"rating": np.tile(np.arange(1, 13), 5), "y": rng.normal(size=60)})
    res = vc.profile_boxes(d, max_levels=12, categorical=["rating"], numeric=["y"])
    assert _ticks(res) == [str(i) for i in range(1, 13)]


def test_profile_boxes_follows_categorical_order():
    lvl = pd.Categorical(["low", "mid", "high"] * 4, categories=["low", "mid", "high", "unused"], ordered=True)
    d = pd.DataFrame({"lvl": lvl, "y": np.arange(12.0)})  # medians: low 4.5, mid 5.5, high 6.5
    res = vc.profile_boxes(d, categorical=["lvl"], numeric=["y"])
    assert _ticks(res) == ["low", "mid", "high"]
    medians = [line.get_ydata()[0] for line in res.figure[0].axes[0].lines if line.get_color() == vc.OKABE_ITO[5]]
    assert medians == [4.5, 5.5, 6.5]


def test_divergent_bar_follows_categorical_order():
    ages = ["0-9", "10-19", "20-29", "30-39"]
    df = pd.DataFrame({"age": pd.Categorical(["20-29", "0-9", "30-39", "10-19", "0-9"], categories=ages, ordered=True),
                       "men": [5, 7, 3, 6, 1], "women": [5, 6, 4, 6, 1]})
    r = vc.divergent_bar(df, "age", "men", "women")
    assert [t.get_text() for t in r.axes.get_yticklabels()] == ages
    assert r.table["age"].tolist() == ages
    assert r.table["men"].tolist() == [8, 6, 5, 3]
    plain = vc.divergent_bar(df.assign(age=df["age"].astype(str)), "age", "men", "women")
    assert plain.table["age"].tolist() == ["20-29", "0-9", "30-39", "10-19"]  # first appearance


# --- missing groups ----------------------------------------------------------------------------------------------


def test_pca_plot_draws_rows_with_a_missing_group_in_grey():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    df["g"] = ["p"] * 10 + ["q"] * 10 + [None] * 10
    r = vc.pca_plot(df, ["a", "b", "c"], group="g")
    assert len(r.table) == sum(len(c.get_offsets()) for c in r.axes.collections) == 30
    assert [t.get_text() for t in r.axes.get_legend().get_texts()] == ["p", "q", "(missing)"]
    grey = r.axes.collections[-1]
    assert np.allclose(grey.get_facecolor()[0][:3], [0x7F / 255] * 3)
    assert np.allclose(grey.get_offsets(), r.table.loc[r.table["g"].isna(), ["PC1", "PC2"]].to_numpy())
    assert list(r.info["ellipses"]) == ["p", "q"]


def test_pca_plot_without_missing_groups_has_no_missing_entry():
    r = vc.pca_plot(datasets.measurements(), ["length", "width", "depth"], group="group")
    assert "(missing)" not in [t.get_text() for t in r.axes.get_legend().get_texts()]


# --- quadrant_plot at extreme magnitudes -------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e160, 1e-170, 1e300])
def test_quadrant_plot_z_scores_do_not_depend_on_the_scale(scale):
    base = pd.DataFrame({"x": [1.0, 2, 3, 4, 6], "y": [1.0, 2, 3, 5, 4]})
    ref = vc.quadrant_plot(base, "x", "y")
    r = vc.quadrant_plot(base.assign(x=base["x"] * scale), "x", "y")
    assert r.table["n"].tolist() == ref.table["n"].tolist()
    assert np.allclose(r.axes.collections[0].get_offsets(), ref.axes.collections[0].get_offsets())
    assert np.allclose(r.info["center"], ref.info["center"], atol=1e-12)


# --- to_pptx ---------------------------------------------------------------------------------------------------


def _slide_titles(path):
    pptx = pytest.importorskip("pptx")
    return [sh.text_frame.text for s in pptx.Presentation(path).slides for sh in s.shapes if sh.has_text_frame]


def test_to_pptx_takes_a_single_string_title_and_any_iterable_of_figures(tmp_path):
    pytest.importorskip("pptx")
    res = vc.waffle(datasets.survey(), category="team")
    assert _slide_titles(vc.to_pptx(res, str(tmp_path / "a.pptx"), titles="Team mix")) == ["Team mix"]
    out = vc.to_pptx((r for r in [res, res.figure]), str(tmp_path / "b.pptx"), titles=iter(["A", "B"]))
    assert _slide_titles(out) == ["A", "B"]


# --- number and level labels -----------------------------------------------------------------------------------


@pytest.mark.parametrize("value, text", [(2.5e18, "2.5e18"), (1e30, "1e30"), (-3e300, "-3e300"), (999.96e12, "1e15"),
                                         (999.94e12, "999.9T"), (12345, "12.3K")])
def test_abbreviate_uses_scientific_notation_beyond_t(value, text):
    assert abbreviate(value) == text


def test_waffle_legend_for_huge_totals_stays_short():
    r = vc.waffle(pd.DataFrame({"c": ["x", "y"], "v": [1e300, 3e300]}), "c", "v")
    assert [t.get_text() for t in r.axes.get_legend().get_texts()] == ["y: 3e300 (75.0%)", "x: 1e300 (25.0%)"]


def test_axis_ticks_beyond_t_use_an_offset_not_thousands_of_t():
    r = vc.dumbbell(pd.DataFrame({"k": ["a", "b"], "s": [1e18, 2e18], "e": [3e18, 4e18]}), "k", "s", "e")
    r.figure.canvas.draw()
    assert not any("T" in t.get_text() for t in r.axes.get_xticklabels())


@pytest.mark.parametrize("value", [pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="+05:00"),
                                   datetime(2024, 1, 1, tzinfo=timezone(timedelta(hours=-3)))])
def test_level_label_shortens_time_zone_aware_midnight(value):
    assert level_label(value) == "2024-01-01"


def test_level_label_keeps_a_time_of_day_with_its_zone():
    assert level_label(pd.Timestamp("2024-01-01 05:00", tz="UTC")) == "2024-01-01 05:00:00+00:00"


def test_percent_grid_facets_read_as_dates_with_a_time_zone():
    d = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01"]).tz_localize("UTC")
    r = vc.percent_grid(pd.DataFrame({"ok": [1, 0, 1], "day": d}), "ok", facet="day")
    assert [a.get_title().split("\n")[0] for a in r.axes.flat] == ["2024-01-01: 50.0%", "2024-02-01: 100.0%"]


# --- exact_mean ------------------------------------------------------------------------------------------------


def _true_mean(v):
    return float(sum(map(Fraction, v)) / len(v))


def test_exact_mean_breaks_a_halfway_tie_with_a_tiny_value():
    v = [1.0, 1.0 + 3 * 2**-52, -1e-300, 0.0]
    assert exact_mean(v) == _true_mean(v) == 0.5000000000000001


def test_exact_mean_is_correctly_rounded_across_wide_ranges():
    rng = random.Random(1)
    for _ in range(3000):
        v = [rng.choice([1, -1]) * rng.random() * 10.0 ** rng.randint(-320, 307) for _ in range(rng.randint(1, 6))]
        v.append(v[0] + 3 * math.ulp(v[0]))
        if rng.random() < 0.3:  # a sum that leaves the float range
            v += [1.7e308, 1.7e308]
        assert exact_mean(v) == _true_mean(v), v


def test_exact_mean_special_values():
    assert math.isnan(exact_mean([]))
    assert math.isnan(exact_mean([math.nan, 1.0]))
    assert math.isnan(exact_mean([math.inf, -math.inf]))
    assert exact_mean([math.inf, 1.0]) == math.inf
    assert exact_mean([1.7e308, 1.7e308, 1e-320]) == _true_mean([1.7e308, 1.7e308, 1e-320])
