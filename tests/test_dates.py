"""Time charts: reading date columns, the range of dates they can draw, date windows and date axes."""

import re
import warnings

import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import timeseries

# Every chart that reads a date column, called with the dates in column "d" (and a copy in "e" for the windows)
_DATE_CALLS = {
    "period_bars": lambda d: vc.period_bars(d, date="d", value="v"),
    "timeseries_fill": lambda d: vc.timeseries_fill(d.assign(w=d["v"].to_numpy()[::-1]), time="d", series=["v", "w"]),
    "calendar_heatmap": lambda d: vc.calendar_heatmap(d, date="d", value="v"),
    "gantt": lambda d: vc.gantt(d.assign(t=range(len(d)), e=d["d"]), task="t", start="d", end="e"),
    "duration_plot": lambda d: vc.duration_plot(d.assign(t=range(len(d)), e=d["d"]), label="t", start="d", end="e",
                                                inner_start="d", inner_end="e"),
}


def _frame(dates, index=None):
    return pd.DataFrame({"d": dates, "v": [float(i) for i in range(len(dates))]}, index=index)


# --- columns that do not parse --------------------------------------------------------------------------------------


@pytest.mark.parametrize("call", list(_DATE_CALLS.values()), ids=list(_DATE_CALLS))
@pytest.mark.parametrize("dates,index,message", [
    # an impossible date in an ISO 8601 column: no advice to use ISO 8601
    (["2024-01-01", "2024-01-05", "2024-13-01", "2024-01-09"], None, "'2024-13-01' (row 2) does not read as a date"),
    (["2024-01-01", "TBD", None], ["a", "b", "c"], "'TBD' (row 'b') does not read as a date"),
    (["2024-01-01", "01/05/2024", "2024-01-07"], None,
     "'01/05/2024' (row 1) is not in the format of the other values; use one consistent format, ideally ISO 8601"),
])
def test_a_column_that_does_not_parse_names_its_first_bad_value(call, dates, index, message):
    with pytest.raises(ValueError, match=re.escape(f"column 'd' could not be parsed as dates: {message}")) as info:
        call(_frame(dates, index))
    assert info.value.__cause__ is not None  # pandas' own error stays in the traceback
    if "format" not in message:
        assert "ISO 8601" not in str(info.value)


@pytest.mark.parametrize("call", list(_DATE_CALLS.values()), ids=list(_DATE_CALLS))
def test_dates_beyond_the_range_of_pandas_2_are_named(call):
    if pd.to_datetime(pd.Series(["9999-12-31"]), errors="coerce").notna().all():
        pytest.skip("pandas 3 holds 9999-12-31")
    with pytest.raises(ValueError, match=re.escape("'9999-12-31' (row 1) is outside the dates pandas")):
        call(_frame(["2024-01-01", "9999-12-31"]))


@pytest.mark.parametrize("name", list(_DATE_CALLS))
@pytest.mark.parametrize("dates,kind", [
    ([True, False], "boolean values (bool)"),
    (pd.to_timedelta([1, 2], unit="D"), "timedelta64 values (timedelta64["),
])
def test_a_column_that_is_not_text_is_named_by_its_type(name, dates, kind):
    if name == "timeseries_fill" and kind.startswith("boolean"):
        return  # timeseries_fill draws a bool time column as numbers
    with pytest.raises(ValueError, match=re.escape(f"column 'd' holds {kind}")) as info:
        _DATE_CALLS[name](_frame(dates))
    assert "ISO 8601" not in str(info.value)


# --- text without a year ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("call", list(_DATE_CALLS.values()), ids=list(_DATE_CALLS))
@pytest.mark.parametrize("dates,first", [
    (["1/2", "1/10", "2/1"], "'1/2' (row 0)"),
    (["Jan 5", "Jan 6", "Feb 1"], "'Jan 5' (row 0)"),
    (["9am", "10am", "11am"], "'9am' (row 0)"),
    (["2024-01-05", "5 Jan", "2024-01-07"], "'5 Jan' (row 1)"),
])
def test_dates_without_a_year_raise_the_same_error_on_every_pandas(call, dates, first):
    # pandas 3 read them as year 1: calendar_heatmap, gantt and duration_plot drew year 0001, the others crashed
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=re.escape(f"column 'd' holds dates without a year, such as {first}; "
                                                       "write the year")):
            call(_frame(dates))
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_text_dates_that_pandas_reads_one_by_one_give_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = vc.period_bars(_frame(["Jan 2024", "Feb 2024", "Mar 2024"]), date="d", value="v")
    assert [str(p) for p in res.table["period"]] == ["2024-01", "2024-02", "2024-03"]
    assert not [str(w.message) for w in caught if "Could not infer format" in str(w.message)]


def test_iso_dates_in_year_1_name_their_year():
    s = pd.Series(["0001-01-05", "0001-02-05"])
    try:
        out = timeseries._dates(s, "d")
    except ValueError as exc:  # pandas 2 holds no dates before 1677
        assert "is outside the dates pandas" in str(exc)
    else:
        assert out.dt.strftime("%Y-%m-%d").tolist() == ["0001-01-05", "0001-02-05"]
