"""Role, infinite-value, constant-metric and missing-group checks in the time series and multivariate charts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets
from viz_calc._core import NEUTRAL

METRICS = ["length", "width", "depth", "mass"]


@pytest.fixture(scope="module")
def meas():
    return datasets.measurements()


# --- one column passed for two roles --------------------------------------------------------------------------------


def test_radar_rejects_a_group_that_is_also_a_metric(meas):
    d = meas.assign(size=meas["length"].round())
    with pytest.raises(ValueError, match="group and metrics must be different columns; both are 'size'"):
        vc.radar(d, metrics=["length", "width", "size"], group="size")
    assert plt.get_fignums() == []


# --- infinite values in numeric roles ------------------------------------------------------------------------------

TIMED = pd.DataFrame({
    "date": pd.date_range("2024-01-29", periods=12, freq="D"),  # spans two months
    "year": np.arange(2001.0, 2013.0),
    "g": list("abc") * 4,
    "v": [3.0, 5.0, 2.0, 8.0, 4.0, 6.0, 7.5, 1.0, 9.0, 2.5, 6.5, 4.5],
    "w": [2.0, 6.0, 1.5, 7.0, 3.0, 8.0, 6.5, 2.0, 8.5, 1.0, 5.5, 3.5],
    "u": [1.0, 4.0, 2.5, 5.0, 2.0, 3.0, 9.5, 6.0, 4.0, 3.5, 7.0, 8.0],
})

INFINITE_CALLS = {
    "period_bars": (lambda d: vc.period_bars(d, date="date", value="v"), "v"),
    "period_bars mean": (lambda d: vc.period_bars(d, date="date", value="v", stat="mean"), "v"),
    "timeseries_fill first": (lambda d: vc.timeseries_fill(d, time="date", series=["v", "w"]), "v"),
    "timeseries_fill second": (lambda d: vc.timeseries_fill(d, time="date", series=["v", "w"]), "w"),
    "timeseries_fill numeric time": (lambda d: vc.timeseries_fill(d, time="year", series=["v", "w"]), "year"),
    "timeseries_fill object time": (
        lambda d: vc.timeseries_fill(d.assign(year=d["year"].astype(object)), time="year", series=["v", "w"]), "year"),
    "timeseries_fill Categorical time": (
        lambda d: vc.timeseries_fill(d.assign(year=d["year"].astype("category")), time="year", series=["v", "w"]),
        "year"),
    "calendar_heatmap": (lambda d: vc.calendar_heatmap(d, date="date", value="v"), "v"),
    "calendar_heatmap max": (lambda d: vc.calendar_heatmap(d, date="date", value="v", stat="max"), "v"),
    "radar": (lambda d: vc.radar(d, metrics=["v", "w", "u"], group="g"), "w"),
    "radar median": (lambda d: vc.radar(d, metrics=["v", "w", "u"], group="g", stat="median"), "w"),
    "pca": (lambda d: vc.pca(d, ["v", "w", "u"]), "u"),
    "pca unscaled": (lambda d: vc.pca(d, ["v", "w", "u"], scale=False), "u"),
    "pca_plot": (lambda d: vc.pca_plot(d, ["v", "w", "u"], group="g"), "v"),
}


def _spoil(data, column, how):
    d = data.copy()
    if how == "all inf":
        d[column] = np.inf
        return d
    d.loc[1, column] = -np.inf if how == "one -inf" else np.inf
    if how == "Float64":
        d[column] = d[column].astype("Float64")
    return d


@pytest.mark.parametrize("how", ["one inf", "one -inf", "all inf", "Float64"])
@pytest.mark.parametrize("name", list(INFINITE_CALLS))
def test_an_infinite_value_raises_naming_the_column(name, how):
    call, column = INFINITE_CALLS[name]
    call(TIMED)  # the clean data draws
    plt.close("all")
    with pytest.raises(ValueError, match=rf"infinite values \(inf or -inf\): \['{column}'\]"):
        call(_spoil(TIMED, column, how))
    assert plt.get_fignums() == []


def test_period_bars_count_and_calendar_heatmap_row_counts_ignore_other_columns():
    d = TIMED.assign(x=np.inf)  # a column in no role
    assert vc.period_bars(d, date="date", value="v", stat="count").table["n"].tolist() == [3, 9]
    assert vc.calendar_heatmap(d, date="date").table["value"].sum() == 12


# --- a constant metric in groups of different sizes ---------------------------------------------------------------


def _unequal_groups(meas):
    return meas.drop(meas.index[meas["group"] == "alpha"][:7])  # 43, 50 and 50 rows


@pytest.mark.parametrize("normalize", ["groups", "data"])
@pytest.mark.parametrize("constant", [0.1, 1 / 3, 0.5])
def test_radar_puts_a_constant_metric_mid_scale_whatever_the_group_sizes(meas, constant, normalize):
    d = _unequal_groups(meas).assign(c=constant)
    t = vc.radar(d, metrics=["length", "width", "c"], group="group", normalize=normalize).table
    assert t["c_mean"].tolist() == [constant] * 3  # exactly, not off by an ulp
    assert t["c_scaled"].tolist() == [0.5] * 3


def test_radar_keeps_the_float_mean_of_groups_that_vary(meas):
    d = _unequal_groups(meas)
    t = vc.radar(d, metrics=["length", "width", "depth"], group="group").table
    expected = d.groupby("group")["length"].mean().reindex(t["group"]).tolist()
    assert t["length_mean"].tolist() == expected


# --- rows with a missing group ---------------------------------------------------------------------------------------


def _without_group(meas, convert=None):
    d = meas.copy()
    if convert is not None:
        d["group"] = convert(d["group"])
    d.loc[[0, 1, 60], "group"] = None
    d.loc[[0, 1, 60], "length"] = [20.0, 30.0, 40.0]  # far above every other length
    return d


@pytest.mark.parametrize("convert", [
    None,
    lambda s: s.astype(object),
    lambda s: s.astype("category"),
    lambda s: s.map({"alpha": 1, "beta": 2, "gamma": 3}).astype("Int64"),
], ids=["as built", "object", "Categorical", "Int64"])
def test_radar_draws_rows_without_a_group_as_missing(meas, convert):
    d = _without_group(meas, convert)
    res = vc.radar(d, metrics=METRICS, group="group")
    t = res.table
    assert len(t) == 4 and t["group"].iloc[:3].notna().all() and pd.isna(t["group"].iloc[3])
    assert t["length_mean"].iloc[3] == 30.0
    assert t["width_mean"].iloc[3] == pytest.approx(d["width"][d["group"].isna()].mean())
    assert t["length_scaled"].iloc[3] == pytest.approx((30.0 - d["length"].min()) / (40.0 - d["length"].min()))
    assert [x.get_text() for x in res.axes.get_legend().get_texts()][-1] == "(missing)"
    assert res.axes.get_lines()[-1].get_color() == NEUTRAL


def test_radar_scales_groups_with_the_missing_group_included(meas):
    t = vc.radar(_without_group(meas), metrics=METRICS, group="group", normalize="groups").table
    assert t["length_scaled"].tolist()[3] == 1.0 and t["length_scaled"].min() == 0.0


def test_radar_scales_data_from_the_rows_drawn_only(meas):
    shown = meas[meas["group"].isin(["alpha", "beta"])]
    t = vc.radar(meas, metrics=METRICS, group="group", order=["beta", "alpha"]).table
    assert t["group"].tolist() == ["beta", "alpha"]
    for m in METRICS:
        lo, hi = shown[m].min(), shown[m].max()
        expected = (shown.groupby("group")[m].mean().reindex(["beta", "alpha"]) - lo) / (hi - lo)
        assert t[f"{m}_scaled"].tolist() == pytest.approx(expected.tolist(), abs=1e-12), m


def test_radar_still_needs_one_named_group(meas):
    with pytest.raises(ValueError, match="no non-missing values"):
        vc.radar(meas.assign(group=None), metrics=METRICS, group="group")
    assert plt.get_fignums() == []
