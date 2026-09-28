"""Regression tests for the thirteenth review round, integration pass: time charts use the shared checks."""

import inspect
import re
import warnings
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import viz_calc as vc
from viz_calc import datasets, timeseries

ROOT = Path(__file__).resolve().parents[1]
DATES = pd.date_range("2024-01-01", periods=6, freq="D")


def _fill_data(**changes):
    return pd.DataFrame({"t": DATES, "a": [1.0, 3.0, 2.0, 5.0, 6.0, 1.0], "b": [2.0] * 6}).assign(**changes)


def _bubble_data(**changes):
    return pd.DataFrame({"t": [1, 1, 2, 2], "x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 1.0, 1.0, 0.0],
                         "s": [1.0, 2.0, 3.0, 4.0], "c": ["a", "b", "a", "b"]}).assign(**changes)


def _duration(proj, **kwargs):
    return vc.duration_plot(proj, label="task", start="start", end="end", inner_start="work_start",
                            inner_end="work_end", **kwargs)


# --- failed calls leave no figure -----------------------------------------------------------------------------------


def test_public_time_functions_clean_up_on_error():
    for name in timeseries.__all__:
        func = getattr(timeseries, name)
        assert hasattr(func, "__wrapped__"), name
        assert inspect.signature(func) == inspect.signature(func.__wrapped__)


def _broken(*args, **kwargs):
    raise RuntimeError("broken after drawing")


@pytest.mark.parametrize("call", [
    lambda: vc.period_bars(pd.DataFrame({"d": DATES, "v": range(6)}), date="d", value="v"),
    lambda: vc.timeseries_fill(_fill_data(), time="t", series=["a", "b"]),
    lambda: vc.gantt(datasets.projects(), task="task", start="start", end="end", group="team"),
    lambda: _duration(datasets.projects()),
])
def test_a_time_chart_that_fails_after_drawing_leaves_no_figure(call, monkeypatch):
    monkeypatch.setattr(timeseries, "_date_axis", _broken)
    with pytest.raises(RuntimeError, match="broken after drawing"):
        call()
    assert plt.get_fignums() == []


def test_calendar_heatmap_and_animated_bubble_that_fail_after_drawing_leave_no_figure(monkeypatch):
    monkeypatch.setattr(matplotlib.figure.Figure, "colorbar", _broken)
    with pytest.raises(RuntimeError, match="broken after drawing"):
        vc.calendar_heatmap(pd.DataFrame({"d": DATES, "v": range(6)}), date="d", value="v")
    monkeypatch.setattr(timeseries, "FuncAnimation", _broken)
    with pytest.raises(RuntimeError, match="broken after drawing"):
        vc.animated_bubble(_bubble_data(), time="t", x="x", y="y", size="s")
    assert plt.get_fignums() == []


def test_a_failed_call_keeps_the_callers_figure():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="valid color"):  # Matplotlib rejects the colour while drawing
        vc.period_bars(pd.DataFrame({"d": DATES, "v": range(6)}), date="d", value="v", color="not a colour", ax=ax)
    assert plt.get_fignums() == [fig.number]


# --- colours for fixed roles ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("colors,match", [
    ("viridis", r"colors must be a tuple of 2 colours \(first series, second series\), got 'viridis'"),
    (("red",), r"colors must be a tuple of 2 colours \(first series, second series\), got 1"),
    (("red", "blue", "green"), "got 3"),
    (("red", "nope"), r"invalid colour\(s\): \['nope'\]"),
])
def test_timeseries_fill_checks_its_colour_pair(colors, match):
    with pytest.raises(ValueError, match=match):
        vc.timeseries_fill(_fill_data(), time="t", series=["a", "b"], colors=colors)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("colors,match", [
    ("viridis", r"colors must be a tuple of 2 colours \(outer, inner\), got 'viridis'"),
    (("r", "g", "b"), r"\(outer, inner\), got 3"),
    (("r", "nope"), r"invalid colour\(s\): \['nope'\]"),
])
def test_duration_plot_checks_its_colour_pair(colors, match):
    with pytest.raises(ValueError, match=match):
        _duration(datasets.projects(), colors=colors)
    assert plt.get_fignums() == []


def test_colour_pairs_are_used_in_their_documented_roles():
    res = vc.timeseries_fill(_fill_data(), time="t", series=["a", "b"], colors=["#111111", "#222222"])
    assert [line.get_color() for line in res.axes.get_lines()] == ["#111111", "#222222"]
    res = _duration(datasets.projects(), colors=["#333333", "#444444"])
    assert [h.get_label() for h in res.axes.get_legend().legend_handles] == ["Overall period", "Active duration"]
    assert [to_hex(h.get_facecolor()) for h in res.axes.get_legend().legend_handles] == ["#333333", "#444444"]


def test_getting_started_lists_the_time_colour_roles():
    text = (ROOT / "docs/getting-started.md").read_text(encoding="utf-8")
    for call, roles in [(lambda: vc.timeseries_fill(_fill_data(), time="t", series=["a", "b"], colors="x"),
                         "`timeseries_fill` (first series, second series)"),
                        (lambda: _duration(datasets.projects(), colors="x"), "`duration_plot` (outer, inner)")]:
        assert roles in text
        with pytest.raises(ValueError, match=re.escape(roles.split(" ", 1)[1])):
            call()


# --- options and counts ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("stat", ["mean", "max", "min"])
def test_calendar_heatmap_stat_without_a_value_column_raises(stat):
    with pytest.raises(ValueError, match=f"stat='{stat}' needs a value column"):
        vc.calendar_heatmap(pd.DataFrame({"d": DATES}), date="d", stat=stat)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("stat", ["sum", "count"])
def test_calendar_heatmap_without_a_value_column_counts_rows(stat):
    t = vc.calendar_heatmap(pd.DataFrame({"d": DATES.repeat(2)}), date="d", stat=stat).table
    assert t["value"].tolist() == [2.0] * 6


@pytest.mark.parametrize("window", [-1, 2.5, "3", True, np.nan])
def test_period_bars_rejects_a_bad_trend_window(window):
    with pytest.raises(ValueError, match=re.escape(f"trend_window must be a whole number of at least 0, got {window!r}")):
        vc.period_bars(pd.DataFrame({"d": DATES, "v": range(6)}), date="d", value="v", trend_window=window)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("window,lines", [(None, 0), (0, 0), (np.int64(2), 1)])
def test_period_bars_trend_window_none_or_zero_draws_no_trend(window, lines):
    res = vc.period_bars(pd.DataFrame({"d": DATES, "v": range(6)}), date="d", value="v", freq="day",
                         trend_window=window)
    assert len(res.axes.get_lines()) == lines
    assert ("trend" in res.table) == bool(lines)


@pytest.mark.parametrize("name,value", [("max_area", -5), ("max_area", 0), ("max_area", np.nan), ("max_area", "big"),
                                        ("interval", 0), ("interval", -100), ("interval", np.inf), ("interval", True)])
def test_animated_bubble_rejects_a_non_positive_area_or_delay(name, value):
    with pytest.raises(ValueError, match=re.escape(f"{name} must be a positive number, got {value!r}")):
        vc.animated_bubble(_bubble_data(), time="t", x="x", y="y", size="s", **{name: value})
    assert plt.get_fignums() == []


def test_animated_bubble_accepts_float_area_and_delay():
    res = vc.animated_bubble(_bubble_data(), time="t", x="x", y="y", size="s", max_area=400.0, interval=250.5)
    assert res.info["area_scale"] == pytest.approx(100)


# --- all-missing columns --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("call,column", [
    (lambda: vc.period_bars(pd.DataFrame({"d": DATES, "v": [np.nan] * 6}), date="d", value="v"), "v"),
    (lambda: vc.period_bars(pd.DataFrame({"d": DATES, "v": pd.array([None] * 6, dtype="Int64")}), date="d",
                            value="v", stat="count"), "v"),
    (lambda: vc.period_bars(pd.DataFrame({"d": [None] * 6, "v": range(6)}), date="d", value="v"), "d"),
    (lambda: vc.timeseries_fill(_fill_data(a=np.nan), time="t", series=["a", "b"]), "a"),
    (lambda: vc.timeseries_fill(_fill_data(t=pd.NaT), time="t", series=["a", "b"]), "t"),
    (lambda: vc.calendar_heatmap(pd.DataFrame({"d": DATES, "v": [np.nan] * 6}), date="d", value="v",
                                 stat="count"), "v"),
    (lambda: vc.calendar_heatmap(pd.DataFrame({"d": [pd.NaT] * 6}), date="d"), "d"),
    (lambda: vc.gantt(datasets.projects().assign(start=pd.NaT), task="task", start="start", end="end"), "start"),
    (lambda: vc.gantt(datasets.projects().assign(end=None), task="task", start="start", end="end"), "end"),
    (lambda: _duration(datasets.projects().assign(end=pd.NaT)), "end"),
    (lambda: vc.animated_bubble(_bubble_data(x=np.nan), time="t", x="x", y="y", size="s"), "x"),
    (lambda: vc.animated_bubble(_bubble_data(y=pd.array([None] * 4, dtype="Float64")), time="t", x="x", y="y",
                                size="s"), "y"),
    (lambda: vc.animated_bubble(_bubble_data(s=np.nan), time="t", x="x", y="y", size="s"), "s"),
])
def test_an_all_missing_column_is_named(call, column):
    with pytest.raises(ValueError, match=f"column '{column}' has no non-missing values"):
        call()
    assert plt.get_fignums() == []


def test_period_bars_values_only_on_undated_rows():
    df = pd.DataFrame({"d": [DATES[0], None], "v": [np.nan, 5.0]})
    with pytest.raises(ValueError, match="no row has both a date in 'd' and a value in 'v'"):
        vc.period_bars(df, date="d", value="v")


def test_duration_plot_still_allows_an_empty_inner_window():
    t = _duration(datasets.projects().assign(work_start=pd.NaT, work_end=pd.NaT)).table
    assert t["inner_days"].isna().all() and t["outer_days"].notna().all()


# --- column lists and legend names ----------------------------------------------------------------------------------


@pytest.mark.parametrize("series", [("a", "b"), pd.Index(["a", "b"]), np.array(["a", "b"]), (c for c in "ab")])
def test_timeseries_fill_accepts_any_list_of_two_columns(series):
    assert vc.timeseries_fill(_fill_data(), time="t", series=series).info["crossovers"] == 2


def test_timeseries_fill_rejects_a_string_or_the_wrong_number_of_series():
    with pytest.raises(TypeError, match=r"series must be a list of column names, not a str; use \['ab'\]"):
        vc.timeseries_fill(_fill_data(ab=1.0), time="t", series="ab")  # not split into 'a' and 'b'
    with pytest.raises(ValueError, match=r"exactly two series, got \['a', 'b', 't'\]"):
        vc.timeseries_fill(_fill_data(), time="t", series=["a", "b", "t"])
    with pytest.raises(TypeError, match="series must be a list of column names, got int"):
        vc.timeseries_fill(_fill_data(), time="t", series=5)


@pytest.mark.parametrize("series", [["a", "a"], ["t", "b"]])
def test_timeseries_fill_needs_three_different_columns(series):
    with pytest.raises(ValueError, match="time and the two series must be three different columns"):
        vc.timeseries_fill(_fill_data(), time="t", series=series)


@pytest.mark.parametrize("labels", ["AB", 7, ["only"], ["x", "y", "z"], []])
def test_timeseries_fill_needs_two_legend_names(labels):
    with pytest.raises(ValueError, match=re.escape(f"labels must be a list of two names, one per series, got {labels!r}")):
        vc.timeseries_fill(_fill_data(), time="t", series=["a", "b"], labels=labels)
    assert plt.get_fignums() == []


def test_timeseries_fill_legend_names_label_the_leader():
    res = vc.timeseries_fill(_fill_data(), time="t", series=["a", "b"], labels=("Alpha", "Beta"))
    assert [t.get_text() for t in res.axes.get_legend().get_texts()] == ["Alpha", "Beta"]
    assert set(res.table["leader"]) == {"Alpha", "Beta"}


# --- date levels read as dates --------------------------------------------------------------------------------------


def test_gantt_legend_and_task_labels_read_dates_without_midnight():
    proj = datasets.projects()
    proj["phase"] = pd.to_datetime(["2024-01-01", "2024-02-01"] * 3)
    proj["milestone"] = pd.Series(list(pd.to_datetime(["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04",
                                                       "2024-03-05", "2024-03-06"])), dtype=object)
    res = vc.gantt(proj, task="milestone", start="start", end="end", group="phase", sort=False)
    assert [t.get_text() for t in res.axes.get_legend().get_texts()] == ["2024-01-01", "2024-02-01"]
    assert [t.get_text() for t in res.axes.get_yticklabels()] == [f"2024-03-0{i}" for i in range(1, 7)]


def test_duration_plot_and_animated_bubble_label_date_levels_as_dates():
    proj = datasets.projects().assign(task=pd.Series(list(pd.date_range("2024-03-01", periods=6)), dtype=object))
    ticks = [t.get_text() for t in _duration(proj, sort=False).axes.get_yticklabels()]
    assert ticks == [f"2024-03-0{i}" for i in range(1, 7)]
    res = vc.animated_bubble(_bubble_data(c=pd.to_datetime(["2024-01-01", "2024-02-01"] * 2)), time="t", x="x",
                             y="y", size="s", color="c")
    assert [t.get_text() for t in res.axes.get_legend().get_texts()] == ["2024-01-01", "2024-02-01"]


# --- gantt colours by group without pandas warnings -----------------------------------------------------------------


def test_gantt_categorical_group_with_unused_categories_does_not_warn():
    proj = datasets.projects()
    teams = ["Ops", "QA", "Unused", "Engineering", "Design", "Research"]
    proj["team"] = pd.Categorical(proj["team"], categories=teams)
    proj.loc[2, "team"] = np.nan  # Engineering still has "Build UI"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # pandas 3 warned (Pandas4Warning) about the unused category
        res = vc.gantt(proj, task="task", start="start", end="end", group="team", sort=False)
    legend = {t.get_text(): to_hex(h.get_facecolor())
              for t, h in zip(res.axes.get_legend().get_texts(), res.axes.get_legend().legend_handles)}
    assert list(legend) == ["Ops", "QA", "Engineering", "Design", "Research", "(missing)"]  # category order, used only
    bars = [to_hex(p.get_facecolor()) for p in res.axes.patches]
    expected = ["(missing)" if pd.isna(t) else t for t in proj["team"]]
    assert bars == [legend[t] for t in expected]


# --- the walkthrough ------------------------------------------------------------------------------------------------


def _blocks(text):
    return [m.group(1) for m in re.finditer(r"^```python\n(.*?)^```", text, re.S | re.M)]


def _table_after(text, marker):
    """The first Markdown table after *marker*, as a list of {column: cell}."""
    lines = text[text.index(marker):].splitlines()
    first = next(i for i, ln in enumerate(lines) if ln.startswith("|"))
    rows = []
    for ln in lines[first:]:
        if not ln.startswith("|"):
            break
        rows.append([c.strip() for c in ln.strip("|").split("|")])
    return [dict(zip(rows[0], r)) for r in rows[2:]]


def test_time_walkthrough_needs_no_ipython_and_shows_what_the_code_prints(tmp_path, monkeypatch):
    text = (ROOT / "docs/walkthroughs/time-and-projects.md").read_text(encoding="utf-8")
    blocks = _blocks(text)
    assert not [ln for b in blocks for ln in b.splitlines() if "IPython" in ln.split("#")[0]]  # comments may name it
    monkeypatch.chdir(tmp_path)  # the animation block saves a GIF
    ns, results = {}, {}
    for code in blocks:
        exec(compile(code, "time-and-projects.md", "exec"), ns)
        for call in ("period_bars", "timeseries_fill", "animated_bubble"):
            if f"vc.{call}(" in code:
                results[call] = ns["res"]
    assert (tmp_path / "bubbles.gif").stat().st_size > 0

    shown = _table_after(text, "vc.period_bars(")
    actual = results["period_bars"].table
    assert len(shown) == 3
    for row, (_, got) in zip(shown, actual.iterrows()):
        assert row["period"] == str(got["period"])
        assert [row[c] for c in ("value", "n", "trend")] == [f"{got[c]:,.0f}" for c in ("value", "n", "trend")]

    crossovers = results["timeseries_fill"].info["crossovers"]
    assert f'res.info["crossovers"]   # {crossovers}' in text

    shown = _table_after(text, "vc.animated_bubble(")
    actual = results["animated_bubble"].table
    assert [r["year"] for r in shown] == [str(y) for y in actual["year"]]  # every frame, in playing order
    for row, (_, got) in zip(shown, actual.iterrows()):
        assert [row[c] for c in ("gdp", "life_exp", "population")] == [
            f"{got[c]:.1f}" for c in ("gdp", "life_exp", "population")]
