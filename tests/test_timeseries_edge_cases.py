"""Time charts: edge cases (nullable values, time order, frame stamps, reversed windows)."""

import gc
import pickle
import warnings
import weakref

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import viz_calc as vc

# --- period_bars -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
@pytest.mark.parametrize("stat", ["sum", "mean", "median", "count", "max", "min"])
def test_period_bars_nullable_values_with_an_empty_period(dtype, stat):
    # February has no rows: a gap for mean/median/max/min, zero for sum/count, exactly as for plain int64
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-03-02", "2024-04-11"]),
                       "v": [10, 20, 30, 40]})
    plain = vc.period_bars(df, date="d", value="v", stat=stat).table
    res = vc.period_bars(df.astype({"v": dtype}), date="d", value="v", stat=stat)
    for col in ("value", "n", "trend"):
        np.testing.assert_allclose(res.table[col].to_numpy(dtype=float), plain[col].to_numpy(dtype=float))
    assert len(res.axes.patches) == 4


def test_period_bars_nullable_period_with_only_missing_values():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-05", "2024-02-02", "2024-03-02"]),
                       "v": pd.array([10, None, 30], dtype="Int64")})
    t = vc.period_bars(df, date="d", value="v", stat="mean").table
    assert t["value"].tolist()[::2] == [10, 30] and np.isnan(t["value"][1])
    assert t["n"].tolist() == [1, 0, 1]


def test_period_bars_n_counts_non_missing_values():
    df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=31), "v": [1.0] * 26 + [np.nan] * 5})
    t = vc.period_bars(df, date="d", value="v", stat="count").table
    assert t["n"].tolist() == [26] and t["value"].tolist() == [26]


# --- calendar_heatmap / gantt options ----------------------------------------------------------------------------


def test_calendar_heatmap_rejects_unknown_stat_before_drawing():
    df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=5), "v": range(5)})
    for bad in ("avg", "median", "std"):
        with pytest.raises(ValueError, match=r"stat must be one of \['sum', 'mean', 'count', 'max', 'min'\]"):
            vc.calendar_heatmap(df, date="d", value="v", stat=bad)
    with pytest.raises(ValueError, match="stat must be one of"):
        vc.calendar_heatmap(df, date="d", stat="avg")  # checked even when rows are counted
    with pytest.raises(ValueError, match="nope"):
        vc.calendar_heatmap(df, date="d", value="v", cmap="nope")
    assert plt.get_fignums() == []


@pytest.mark.parametrize("today", ["nope", "", "Now", object()])
def test_gantt_rejects_a_bad_today_before_drawing(today):
    df = pd.DataFrame({"t": ["x"], "s": ["2024-01-01"], "e": ["2024-02-01"]})
    with pytest.raises(ValueError, match="today must be a date or 'now'"):
        vc.gantt(df, task="t", start="s", end="e", today=today)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("today", ["now", "2024-01-15", pd.Timestamp("2024-01-15"), np.datetime64("2024-01-15")])
def test_gantt_accepts_dates_and_now_for_today(today):
    df = pd.DataFrame({"t": ["x"], "s": ["2024-01-01"], "e": ["2024-02-01"]})
    res = vc.gantt(df, task="t", start="s", end="e", today=today)
    assert len(res.axes.lines) == 1


# --- duration_plot -----------------------------------------------------------------------------------------------


def test_duration_plot_rejects_reversed_windows():
    d = pd.DataFrame({"lab": ["x", "y"], "s": ["2024-01-10", "2024-01-01"], "e": ["2024-01-01", "2024-01-31"],
                      "is": ["2024-01-02", "2024-01-05"], "ie": ["2024-01-05", "2024-01-10"]})
    with pytest.raises(ValueError, match=r"'e' is before 's' for row\(s\): \['x'\]"):
        vc.duration_plot(d, "lab", "s", "e", "is", "ie")
    d2 = pd.DataFrame({"lab": ["z"], "s": ["2024-01-01"], "e": ["2024-01-31"], "is": ["2024-01-20"],
                       "ie": ["2024-01-10"]})
    with pytest.raises(ValueError, match=r"'ie' is before 'is' for row\(s\): \['z'\]"):
        vc.duration_plot(d2, "lab", "s", "e", "is", "ie")
    ok = d2.assign(ie=["2024-01-25"])
    assert vc.duration_plot(ok, "lab", "s", "e", "is", "ie").table["inner_share"].tolist() == [5 / 30]


# --- mixed UTC offsets in a Categorical --------------------------------------------------------------------------


@pytest.mark.parametrize("call", [
    lambda d: vc.gantt(d, "t", "s", "e"),
    lambda d: vc.duration_plot(d, "t", "s", "e", "s", "e"),
    lambda d: vc.timeseries_fill(d.assign(a=[1.0, 2.0], b=[2.0, 1.0]), time="s", series=["a", "b"]),
])
def test_categorical_mixing_offset_and_offsetless_dates_gives_the_clear_error(call):
    d = pd.DataFrame({"t": ["a", "b"], "s": ["2024-01-01T00:00+01:00", "2024-01-02"],
                      "e": ["2024-02-01T00:00+01:00", "2024-02-01T00:00+01:00"]})
    with pytest.raises(ValueError, match="mixes values with and without a UTC offset"):
        call(d.astype({"s": "category"}))


# --- animated_bubble ---------------------------------------------------------------------------------------------


def _bubbles(t):
    n = len(t)
    df = pd.DataFrame({"t": t, "x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float),
                       "s": np.arange(1, n + 1, dtype=float)})
    return vc.animated_bubble(df, time="t", x="x", y="y", size="s")


def _drawn_stamps(res):
    anim = res.info["animation"]
    out = []
    for i in range(len(res.info["frames"])):
        anim._func(i)  # draw frame i as the animation would
        out.append(res.axes.texts[-1].get_text())
    return out


@pytest.mark.parametrize("t,frames", [
    (["10/1/2020", "9/1/2020", "10/1/2020", "12/1/2020"], ["9/1/2020", "10/1/2020", "12/1/2020"]),
    (["10", "2", "1", "2"], ["1", "2", "10"]),
    (["2024-01-01T10:00+09:00", "2024-01-01T04:00+00:00"], ["2024-01-01T10:00+09:00", "2024-01-01T04:00+00:00"]),
    (pd.Series(["10/1/2020", "9/1/2020"] * 2).astype("category"), ["9/1/2020", "10/1/2020"]),
    (pd.Categorical(["early", "late"] * 2, categories=["late", "early"], ordered=True), ["late", "early"]),
    (pd.Categorical(["10/1/2020", "9/1/2020"], categories=["10/1/2020", "9/1/2020"], ordered=True),
     ["10/1/2020", "9/1/2020"]),
    (pd.Categorical(["Q1 2020", "Q4 2019"] * 2, categories=["Q4 2019", "Q1 2020"], ordered=True),
     ["Q4 2019", "Q1 2020"]),
    (["Phase B", "Phase A", "Phase C"], ["Phase A", "Phase B", "Phase C"]),
])
def test_animated_bubble_plays_frames_in_time_order(t, frames):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no pandas date-format hint for labels that are not dates
        res = _bubbles(t)
    assert list(res.info["frames"]) == frames
    assert _drawn_stamps(res) == frames  # the animation plays them in that order, labelled as written
    assert res.table["t"].tolist() == frames  # and the table follows the same order


def test_animated_bubble_ordered_categorical_table_matches_frames():
    months = pd.Categorical(["Mar", "Jan", "Apr", "Feb"], categories=["Jan", "Feb", "Mar", "Apr"], ordered=True)
    res = _bubbles(months)
    assert list(res.info["frames"]) == ["Jan", "Feb", "Mar", "Apr"] == res.table["t"].tolist()


@pytest.mark.parametrize("t,stamps", [
    (pd.to_datetime(["2020-02-01", "2020-01-01"] * 2), ["2020-01-01", "2020-02-01"]),
    (pd.to_datetime(["2020-01-01 10:00", "2020-01-01 09:00"] * 2).tz_localize("UTC"),
     ["2020-01-01 09:00 UTC", "2020-01-01 10:00 UTC"]),
    (pd.to_datetime(["2024-11-03 05:30", "2024-11-03 06:30"]).tz_localize("UTC").tz_convert("America/New_York"),
     ["2024-11-03 01:30 EDT", "2024-11-03 01:30 EST"]),  # the repeated DST hour stays distinguishable
    (pd.to_datetime(["2020-01-01 00:00:05", "2020-01-01 00:00:00"]), ["2020-01-01 00:00:00", "2020-01-01 00:00:05"]),
    ([2000, None, 2001, 2001], ["2000", "2001"]),
    ([0.5, 1.5], ["0.5", "1.5"]),
    (pd.to_timedelta([2, 1], unit="D"), ["1 days 00:00:00", "2 days 00:00:00"]),
])
def test_animated_bubble_frame_stamps_are_readable(t, stamps):
    assert _drawn_stamps(_bubbles(t)) == stamps


def test_animated_bubble_rejects_an_all_missing_time_column():
    for t in ([np.nan] * 3, pd.Categorical([None] * 3, categories=["a"])):
        with pytest.raises(ValueError, match="'t' has no non-missing values"):
            _bubbles(t)


def test_animated_bubble_unused_animation_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = _bubbles([1, 1, 2, 2]).table  # only the numbers are wanted
        plt.close("all")
        gc.collect()
    assert not [w for w in caught if "Animation was deleted" in str(w.message)]


def test_animated_bubble_animation_lives_as_long_as_its_figure():
    res = _bubbles([1, 1, 2, 2])
    anim, fig = weakref.ref(res.info["animation"]), res.figure
    del res
    gc.collect()
    assert anim() is not None  # plt.show() would still play it
    assert "<script" in anim().to_jshtml()
    assert isinstance(pickle.loads(pickle.dumps(fig)), plt.Figure)  # the figure still pickles
    plt.close(fig)
    del fig
    gc.collect()
    assert anim() is None  # and nothing leaks once the figure is gone
