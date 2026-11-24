"""Aggregates in charts: integer widths, float16, nullable and Arrow columns, overflow and correctly rounded means."""

import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets


def _render(res):
    res.figure.savefig(io.BytesIO(), format="png")  # some failures only appear when the figure is drawn


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# --- float16 sums (pandas 3 keeps float16, which overflows past 65,504) -------------------------------------------


_F16 = np.array([40000, 40000, 1], dtype="float16")


@pytest.mark.parametrize("draw, column", [
    (lambda: vc.lollipop(pd.DataFrame({"k": ["x", "x", "y"], "a": _F16}), x="k", y="a", stat="sum"), "value"),
    (lambda: vc.period_bars(pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01"]),
                                          "a": _F16}), date="d", value="a"), "value"),
    (lambda: vc.waffle(pd.DataFrame({"k": ["x", "x", "y"], "a": _F16}), category="k", value="a"), "value"),
    (lambda: vc.divergent_bar(pd.DataFrame({"k": ["x", "x", "y"], "a": _F16, "b": _F16}), "k", "a", "b"), "a"),
], ids=["lollipop", "period_bars", "waffle", "divergent_bar"])
def test_float16_values_are_summed_in_float64(draw, column):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the float16 sum also warned "overflow encountered in reduceat"
        res = draw()
    assert sorted(res.table[column].tolist()) == [1.0, 80000.0]
    _render(res)


def test_sankey_sums_float16_flows_in_float64():
    pytest.importorskip("plotly")
    res = vc.sankey(pd.DataFrame({"a": ["x", "x", "y"], "b": ["y", "y", "z"], "v": _F16}), "a", "b", "v")
    assert res.table["outflow"].tolist() == [80000.0, 1.0, 0.0]


def test_ridgeplot_mean_of_float16_does_not_overflow():
    # pandas 2.0 took the mean of a float16 Series in float16: inf for values near 60,000.
    d = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 2, "y": np.array([60000, 65000, 64000, 61000, 1, 2], dtype="float16")})
    t = vc.ridgeplot(d, x="y", group="g").table
    assert t["mean"].tolist() == [62496.0, 1.5]


# --- integer widths and bool --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["uint8", "uint32", "int8"])
def test_divergent_bar_draws_left_bars_of_small_and_unsigned_integers(dtype):
    res = vc.divergent_bar(pd.DataFrame({"k": ["a", "b"], "l": np.array([100, 10], dtype=dtype),
                                         "r": np.array([5, 7], dtype=dtype)}), "k", "l", "r")
    assert [p.get_width() for p in res.axes.patches] == [-100, -10, 5, 7]  # uint8 drew 156 and 246
    _render(res)


def test_divergent_bar_sums_a_repeated_unsigned_category():
    res = vc.divergent_bar(pd.DataFrame({"k": ["a", "a", "b"], "l": np.array([200, 100, 10], dtype="uint8"),
                                         "r": np.array([5, 7, 1], dtype="uint8")}), "k", "l", "r")
    assert res.table["l"].tolist() == [300, 10]  # pandas 3 raised OverflowError negating the uint64 sums
    assert [p.get_width() for p in res.axes.patches][:2] == [-300, -10]


def test_sankey_totals_of_large_integer_flows_do_not_wrap_around():
    pytest.importorskip("plotly")
    res = vc.sankey(pd.DataFrame({"a": ["x", "x", "y"], "b": ["y", "y", "z"], "v": np.array([2**62] * 3)}),
                    "a", "b", "v")
    assert res.table["inflow"].tolist() == [0.0, 2.0**63, 2.0**62]  # int64 gave -2**63 for node y
    assert res.table["outflow"].tolist() == [2.0**63, 2.0**62, 0.0]


def test_ridgeplot_quartiles_of_int8_do_not_wrap_around():
    y = np.array([-128, 127, 0, -5, 100, -100, 3, 7], dtype="int8")
    t = vc.ridgeplot(pd.DataFrame({"g": list("abababab"), "y": y}), x="y", group="g").table
    expected = vc.ridgeplot(pd.DataFrame({"g": list("abababab"), "y": y.astype(float)}), x="y", group="g").table
    assert t["q1"].tolist() == expected["q1"].tolist() == [-32.0, -28.75]  # int8 interpolation gave +32


def test_profile_boxes_draws_int8_quartiles_without_wrapping():
    d = pd.DataFrame({"g": ["a"] * 4, "y": np.array([-128, 0, 3, 100], dtype="int8")})
    res = vc.profile_boxes(d, categorical=["g"], numeric=["y"])
    box = res.figure[0].axes[0].lines[0]  # the lower whisker runs from the first quartile down
    assert box.get_ydata()[0] == np.quantile([-128.0, 0.0, 3.0, 100.0], 0.25) == -32.0


def test_bool_columns_are_summarised_as_zero_and_one():
    d = pd.DataFrame({"g": list("abababab"), "y": [True, False, True, True, False, False, True, False]})
    t = vc.ridgeplot(d, x="y", group="g").table  # NumPy cannot subtract booleans in the quartiles
    assert t["mean"].tolist() == [0.75, 0.25] and t["q3"].tolist() == [1.0, 0.25]
    meas = datasets.measurements()
    res = vc.profile_boxes(meas.assign(b=np.tile([True, False], len(meas) // 2)), categorical=["group"],
                           numeric=["b"])
    assert res.table["numeric"].tolist() == ["b"]


# --- sums beyond the largest float ---------------------------------------------------------------------------------


def test_sankey_reports_flows_that_add_up_beyond_the_largest_float():
    pytest.importorskip("plotly")
    with pytest.raises(ValueError, match=r"the 'v' values of \('x', 'y'\) add up to more than the largest float"):
        vc.sankey(pd.DataFrame({"a": ["x", "x"], "b": ["y", "y"], "v": [1e308, 1e308]}), "a", "b", "v")


@pytest.mark.parametrize("draw", [
    lambda d: vc.waffle(d, "k", "v"),
    lambda d: vc.divergent_bar(d.assign(w=1.0), "k", "v", "w"),
    lambda d: vc.lollipop(d, "k", "v", stat="sum"),
], ids=["waffle", "divergent_bar", "lollipop"])
def test_category_sums_beyond_the_largest_float_name_the_column(draw):
    d = pd.DataFrame({"k": ["x", "x", "y"], "v": [1e308, 1e308, 1.0]})
    with pytest.raises(ValueError, match=r"the 'v' values of 'x' add up to more than the largest float"):
        draw(d)


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")  # Matplotlib's own box statistics
def test_summary_mean_of_values_whose_sum_overflows_is_finite():
    y = np.random.default_rng(0).uniform(1, 2, 200) * 1e306  # the sum is beyond the largest float, the mean is not
    d = pd.DataFrame({"g": ["a"] * 200, "y": y})
    expected = float(np.mean(y / 2.0**1000) * 2.0**1000)
    for res in (vc.ridgeplot(d, x="y", group="g"), vc.raincloud(d, x="g", y="y"), vc.histogram(d, x="y")):
        assert res.table["mean"].tolist() == [expected]  # inf before


# --- extreme magnitudes ------------------------------------------------------------------------------------------

_FEATURES = ["length", "width", "depth", "mass"]


@pytest.mark.parametrize("factor", [1e300, 1e-200])
@pytest.mark.parametrize("scale", [True, False])
def test_pca_does_not_depend_on_the_magnitude_of_the_data(factor, scale):
    meas = datasets.measurements()
    expected = vc.pca(meas, _FEATURES, scale=scale)["explained_variance_ratio"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # x 1e300 overflowed to nan; x 1e-200 read every feature as constant
        got = vc.pca(meas.assign(**{c: meas[c] * factor for c in _FEATURES}), _FEATURES, scale=scale)
    assert got["explained_variance_ratio"].tolist() == pytest.approx(expected.tolist(), rel=1e-9)


def test_pearson_r_of_values_near_the_float_limit():
    meas = datasets.measurements()
    expected = vc.stats.correlation_test(meas["length"], meas["width"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = vc.stats.correlation_test(meas["length"] * 2.0**1000, meas["width"])  # nan, or a SciPy error
        table = vc.correlogram(meas.assign(length=meas["length"] * 1e306)[_FEATURES]).table
    assert got == expected  # a power of two changes no digit
    assert np.isfinite(table["r"]).all()


def test_bootstrap_ci_of_values_near_the_float_limit():
    a, b = np.array([1.0, 2.0, 4.0, 7.0]), np.array([3.0, 5.0, 6.0, 9.0, 10.0])
    for samples in ((a,), (a, b)):
        expected = vc.stats.bootstrap_ci(*samples)
        for factor in (2.0**1000, 2.0**-1000):  # BCa's power sums overflowed or underflowed: nan limits
            got = vc.stats.bootstrap_ci(*(s * factor for s in samples))
            assert got == tuple(v * factor for v in expected)


@pytest.mark.parametrize("factor", [1e-200, 1e39])
def test_pies_draw_values_beyond_float32(factor):
    # Matplotlib 3.7 casts wedge sizes to float32, which gave 0 / 0 or inf / inf and a crash
    donut = vc.donut_grid(pd.DataFrame({"a": [3.0, 5.0], "b": [4.0, 1.0]}) * factor)
    assert donut.table["percent"].round(6).tolist() == [42.857143, 83.333333, 57.142857, 16.666667]
    nested = vc.nested_pie(pd.DataFrame({"p": list("aab"), "c": list("xyz"), "v": [1.0, 2.0, 3.0]}).assign(
        v=lambda d: d["v"] * factor), outer="p", inner="c", value="v")
    assert nested.table["percent_of_total"].round(6).tolist() == [16.666667, 33.333333, 50.0]
    _render(donut)
    _render(nested)


def test_donut_grid_names_a_row_whose_values_overflow():
    with pytest.raises(ValueError, match=r"the values of row 'r' add up to more than the largest float"):
        vc.donut_grid(pd.DataFrame({"a": [1e308], "b": [1e308]}, index=["r"]))


def test_quadrant_plot_of_values_near_the_float_limit_does_not_warn():
    meas = datasets.measurements()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the tie tolerance took np.mean(np.abs(x)), which overflowed
        res = vc.quadrant_plot(meas.assign(length=meas["length"] * 1e306), x="length", y="width")
    assert np.isfinite(res.info["correlation"]["r"])


# --- missing values in Arrow-backed columns -----------------------------------------------------------------------


def test_divergent_bar_keeps_an_all_missing_arrow_category_missing():
    pytest.importorskip("pyarrow")
    d = pd.DataFrame({"k": ["a", "a", "b", "c"], "l": [1, 2, None, 4], "r": [1, 2, 3, None]})
    t = vc.divergent_bar(d.convert_dtypes(dtype_backend="pyarrow"), "k", "l", "r").table
    # pandas 2.0 ignores min_count for Arrow values, so the categories without a value read 0
    assert t["l"].tolist()[:1] + t["l"].tolist()[2:] == [3.0, 4.0] and np.isnan(t["l"].iloc[1])
    assert np.isnan(t["r"].iloc[2]) and t["l"].dtype == t["r"].dtype == np.float64


# --- nullable columns in tables -----------------------------------------------------------------------------------


def _trial_with_an_empty_arm():
    t = datasets.trial()
    t["score"] = pd.array(t["score"].round(), dtype="Int64")
    t.loc[t["arm"] == "low dose", "score"] = pd.NA
    return t


@pytest.mark.parametrize("stat", ["mean", "median"])
def test_lollipop_draws_a_nullable_group_with_no_values(stat):
    res = vc.lollipop(_trial_with_an_empty_arm(), x="arm", y="score", stat=stat)  # float(pd.NA) failed
    assert res.table["value"].dtype == np.float64 and res.table["value"].isna().sum() == 1
    _render(res)


def test_lollipop_counts_of_a_nullable_column_are_int64():
    res = vc.lollipop(_trial_with_an_empty_arm(), x="arm", y="score", stat="count")
    assert res.table["value"].dtype == np.int64  # Int64 on pandas 3


def test_bullet_table_holds_nullable_values_as_floats():
    df = pd.DataFrame({"k": list("wxyz"), "a": pd.array([3, None, 2, 8], dtype="Int64"), "b": [5, 5, 5, 5]})
    t = vc.bullet(df, label="k", value="a", target="b").table
    assert t["value"].dtype == np.float64 and np.isnan(t["value"].iloc[1])  # object with pd.NA before
    assert vc.bullet(df.fillna({"a": 4}), label="k", value="a", target="b").table["value"].dtype == np.int64


def test_bullet_table_converts_arrow_values():
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"k": list("wxy"), "a": [3, None, 2], "b": [5.0, 5.5, 5.0]})
    t = vc.bullet(df.convert_dtypes(dtype_backend="pyarrow"), label="k", value="a", target="b").table
    assert t["value"].dtype == np.float64 and t["target"].dtype == np.float64


def test_ridgeplot_summary_of_a_nullable_group_with_no_values_is_float():
    d = pd.DataFrame({"g": list("aaabbb"), "y": pd.array([1.0, 2.0, 3.0, None, None, None], dtype="Float64")})
    t = vc.ridgeplot(d, x="y", group="g").table
    assert (t[["mean", "median", "sd", "q1", "q3"]].dtypes == np.float64).all()  # object with pd.NA before
    assert t["n"].tolist() == [3, 0] and t["mean"].iloc[1] != t["mean"].iloc[1]


# --- correctly rounded group means (independent of row order and pandas version) ----------------------------------


def test_lollipop_does_not_fade_a_mean_equal_to_the_threshold():
    res = vc.lollipop(pd.DataFrame({"g": ["b"] * 4, "y": [9.2, 6.0, 8.8, 9.6]}), x="g", y="y", threshold=8.4)
    assert res.table["value"].tolist() == [8.4]  # 8.399999999999999 on pandas 3, faded as below
    assert res.axes.collections[-1].get_alpha() == 1.0


def test_period_bars_mean_does_not_depend_on_row_order():
    d = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 3 + ["2024-02-01"] * 3),
                      "v": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1]})
    assert vc.period_bars(d, date="date", value="v", stat="mean").table["value"].tolist() == [0.2, 0.2]


def test_calendar_heatmap_mean_does_not_depend_on_row_order():
    d = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3),
                      "v": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1]})
    t = vc.calendar_heatmap(d, date="date", value="v", stat="mean").table
    assert t["value"].dropna().tolist() == [0.2, 0.2]


def test_radar_groups_with_equal_means_sit_mid_scale():
    d = pd.DataFrame({"grp": ["a"] * 3 + ["b"] * 3, "m1": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
                      "m2": [1.0, 2, 3, 4, 5, 6], "m3": [3.0, 2, 1, 6, 5, 4]})
    t = vc.radar(d, metrics=["m1", "m2", "m3"], group="grp", normalize="groups").table
    assert t["m1_mean"].tolist() == [0.2, 0.2] and t["m1_scaled"].tolist() == [0.5, 0.5]  # 0 and 1 on pandas 3


def test_timeseries_fill_averages_repeated_stamps_independently_of_row_order():
    d = pd.DataFrame({"t": [1, 1, 1, 2, 2, 2], "a": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
                      "b": [0.3, 0.2, 0.1, 0.1, 0.2, 0.3]})
    res = vc.timeseries_fill(d, time="t", series=["a", "b"])
    assert res.table["difference"].tolist() == [0.0, 0.0] and res.info["crossovers"] == 0


def test_animated_bubble_table_means_do_not_depend_on_row_order():
    d = pd.DataFrame({"t": [1, 1, 1, 2, 2, 2], "x": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1], "y": [1.0] * 6, "s": [1.0] * 6})
    t = vc.animated_bubble(d, time="t", x="x", y="y", size="s").table
    assert t["x"].tolist() == [0.2, 0.2]
