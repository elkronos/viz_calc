"""Role, infinite-value, constant-metric and missing-group checks in the time series and multivariate charts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets


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
