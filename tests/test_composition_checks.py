"""Part-to-whole charts: input validation, colour roles, date labels and figure cleanup."""

import inspect
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import viz_calc as vc
from viz_calc import composition, datasets

WIDE = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["r1", "r2"])
STEPS = pd.DataFrame({"l": list("abc"), "v": [10, -4, 3]})
GAINS = STEPS.assign(v=STEPS["v"].abs())


# --- figures are closed when a call fails ---------------------------------------------------------------------------


def test_public_plotting_functions_clean_up_on_error():
    for name in composition.__all__:
        func = getattr(composition, name)
        assert hasattr(func, "__wrapped__"), name
        assert inspect.signature(func) == inspect.signature(func.__wrapped__)


def _broken(*args, **kwargs):
    raise RuntimeError("failed while drawing")


@pytest.mark.parametrize(("patch", "call"), [
    ("abbreviate", lambda: vc.waffle(datasets.trial(), category="arm")),
    ("st.wilson_ci", lambda: vc.percent_grid(datasets.trial(), column="improved", facet="arm")),
    ("text_color", lambda: vc.stacked_percentages(datasets.trial(), group="site", category="arm")),
    ("abbreviate", lambda: vc.donut_grid(WIDE)),
    ("text_color", lambda: vc.nested_pie(datasets.trial(), outer="arm", inner="site")),
    ("abbreviate", lambda: vc.circular_bar(GAINS, label="l", value="v")),
    ("abbreviate", lambda: vc.waterfall(STEPS, label="l", value="v")),
    ("abbreviate", lambda: vc.funnel(GAINS, stage="l", value="v")),
    ("level_label", lambda: vc.bullet(STEPS, label="l", value="v")),
    ("NEUTRAL", lambda: vc.upset({"A": {1, 2}, "B": {2, 3}})),
], ids=composition.__all__)
def test_a_call_that_fails_after_drawing_leaves_no_figure(monkeypatch, patch, call):
    owner, name = (composition.st, "wilson_ci") if patch.startswith("st.") else (composition, patch)
    monkeypatch.setattr(owner, name, "not a colour" if name == "NEUTRAL" else _broken)
    with pytest.raises((RuntimeError, ValueError)):
        call()
    assert plt.get_fignums() == []


def test_a_failed_call_leaves_the_callers_figure_open(monkeypatch):
    monkeypatch.setattr(composition, "level_label", _broken)
    fig, ax = plt.subplots()
    with pytest.raises(RuntimeError):
        vc.bullet(STEPS.head(1), label="l", value="v", ax=ax)
    assert plt.get_fignums() == [fig.number]


# --- fixed-role colours ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("colors", "message"), [
    ("viridis", r"colors must be a tuple of 3 colours \(increase, decrease, total\), got 'viridis'"),
    (("red", "blue"), r"colors must be a tuple of 3 colours \(increase, decrease, total\), got 2"),
    (("red", "blue", "green", "black"), r"colors must be a tuple of 3 colours \(increase, decrease, total\), got 4"),
    (("red", "blue", "x"), r"colors contains invalid colour\(s\): \['x'\]"),
])
def test_waterfall_colours_are_three_roles(colors, message):
    with pytest.raises(ValueError, match=message):
        vc.waterfall(STEPS, label="l", value="v", colors=colors)
    assert plt.get_fignums() == []


def test_waterfall_colours_follow_the_role_order():
    res = vc.waterfall(STEPS, label="l", value="v", colors=["#111111", "#222222", (0.2, 0.2, 0.2)])
    assert [to_hex(p.get_facecolor()) for p in res.axes.patches] == ["#111111", "#222222", "#111111", "#333333"]


@pytest.mark.parametrize("colors", ["viridis", ("red",), ("red", "blue", "green")])
def test_percent_grid_colours_are_two_roles(colors):
    with pytest.raises(ValueError, match=r"colors must be a tuple of 2 colours \(success, other\)"):
        vc.percent_grid(datasets.trial(), column="improved", colors=colors)
    assert plt.get_fignums() == []


def test_percent_grid_accepts_any_colour_form():
    data = pd.DataFrame({"y": [1, 0, 0, 0]})
    res = vc.percent_grid(data, column="y", colors=[(1.0, 0.0, 0.0), "#0000ff"])
    dots = [to_hex(c) for c in res.axes[0, 0].collections[0].get_facecolors()]
    assert dots.count("#ff0000") == 25 and dots.count("#0000ff") == 75


# --- options and counts ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"max_intersections": True}, "max_intersections must be an integer >= 1, got True"),
    ({"max_intersections": 0}, "max_intersections must be an integer >= 1, got 0"),
    ({"min_size": "2"}, "min_size must be a finite number, got '2'"),
    ({"min_size": None}, "min_size must be a finite number, got None"),
    ({"min_size": np.nan}, "min_size must be a finite number"),
])
def test_upset_counts_are_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        vc.upset({"A": {1, 2}, "B": {2, 3}}, **kwargs)
    assert plt.get_fignums() == []


def test_upset_min_size_may_be_any_number():
    sets = {"A": {1, 2, 3}, "B": {3, 4}}
    assert vc.upset(sets, min_size=1.5).table["size"].tolist() == [2]
    assert vc.upset(sets, min_size=0, max_intersections=None).table["size"].sum() == 4


# --- all-missing columns --------------------------------------------------------------------------------------------

NAN = np.nan


@pytest.mark.parametrize(("column", "call"), [
    ("c", lambda: vc.waffle(pd.DataFrame({"c": [NAN, NAN]}), "c")),
    ("v", lambda: vc.waffle(pd.DataFrame({"c": ["a", "b"], "v": [NAN, NAN]}), "c", "v")),
    ("y", lambda: vc.percent_grid(pd.DataFrame({"y": [NAN, NAN]}), "y")),
    ("o", lambda: vc.nested_pie(pd.DataFrame({"o": [NAN, NAN], "i": ["x", "y"]}), "o", "i")),
    ("i", lambda: vc.nested_pie(pd.DataFrame({"o": ["a", "b"], "i": [None, None]}), "o", "i")),
    ("l", lambda: vc.circular_bar(pd.DataFrame({"l": [NAN, NAN], "v": [1, 2]}), "l", "v")),
    ("v", lambda: vc.circular_bar(pd.DataFrame({"l": ["a", "b"], "v": [NAN, NAN]}), "l", "v")),
    ("v", lambda: vc.circular_bar(pd.DataFrame({"l": ["a", "b"], "v": pd.array([None, None], dtype="Int64")}),
                                  "l", "v")),
    ("l", lambda: vc.waterfall(pd.DataFrame({"l": [NAN, NAN], "v": [1.0, 2.0]}), "l", "v")),
    ("s", lambda: vc.funnel(pd.DataFrame({"s": [NAN, NAN], "v": [2, 1]}), "s", "v")),
    ("v", lambda: vc.funnel(pd.DataFrame({"s": ["a", "b"], "v": [NAN, NAN]}), "s", "v")),
    ("k", lambda: vc.bullet(pd.DataFrame({"k": [NAN], "v": [1.0]}), "k", "v")),
    ("v", lambda: vc.bullet(pd.DataFrame({"k": ["a", "b"], "v": [NAN, NAN]}), "k", "v", bands=[1, 2])),
])
def test_an_all_missing_column_is_named(column, call):
    with pytest.raises(ValueError, match=f"column '{column}' has no non-missing values"):
        call()
    assert plt.get_fignums() == []


@pytest.mark.parametrize("sets", [pd.DataFrame({"A": [], "B": []}), {"A": [], "B": set()},
                                  pd.DataFrame({"A": [NAN, False], "B": [None, 0]})],
                         ids=["no rows", "empty sets", "no members"])
def test_upset_needs_a_member(sets):
    with pytest.raises(ValueError, match="upset needs at least one member; every set is empty"):
        vc.upset(sets)
    assert plt.get_fignums() == []


# --- column lists ---------------------------------------------------------------------------------------------------


def test_donut_grid_columns_is_not_split_into_characters():
    with pytest.raises(TypeError, match=r"columns must be a list of column names, not a str; use \['ab'\]"):
        vc.donut_grid(WIDE.assign(ab=[1, 1]), columns="ab")


def test_bullet_bands_is_not_split_into_characters():
    data = pd.DataFrame({"k": ["x"], "v": [5], "poor": [3], "p": [1], "o": [2], "r": [4]})
    with pytest.raises(TypeError, match=r"bands must be a list of column names, not a str; use \['poor'\]"):
        vc.bullet(data, "k", "v", bands="poor")
    assert len(vc.bullet(data, "k", "v", bands=pd.Index(["poor"])).axes[0].patches) == 2  # one band, the measure


# --- datetime levels read as dates ----------------------------------------------------------------------------------

MONTHS = pd.DataFrame({"m": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01"]), "c": ["a", "b", "a"],
                       "y": [1, 0, 1], "v": [1.0, 2.0, 3.0]})


def _all_text(res):
    fig = res.figure
    return [t.get_text() for t in fig.findobj(lambda a: hasattr(a, "get_text")) if t.get_text()]


@pytest.mark.parametrize("call", [
    lambda: vc.stacked_percentages(MONTHS, group="m", category="c"),
    lambda: vc.stacked_percentages(MONTHS, group="c", category="m"),
    lambda: vc.percent_grid(MONTHS, column="y", facet="m"),
    lambda: vc.waffle(MONTHS, category="m"),
    lambda: vc.donut_grid(MONTHS.pivot_table(index="m", columns="c", values="v", aggfunc="sum", fill_value=0)),
    lambda: vc.donut_grid(MONTHS.pivot_table(index="c", columns="m", values="v", aggfunc="sum", fill_value=0)),
    lambda: vc.nested_pie(MONTHS, outer="m", inner="c"),
    lambda: vc.nested_pie(MONTHS, outer="c", inner="m"),
    lambda: vc.circular_bar(MONTHS, label="m", value="v", group="m"),
    lambda: vc.bullet(MONTHS.head(1), label="m", value="v"),
], ids=["stacked ticks", "stacked legend", "percent_grid", "waffle", "donut titles", "donut legend",
        "nested outer", "nested inner", "circular_bar", "bullet"])
def test_datetime_levels_read_as_dates(call):
    texts = _all_text(call())
    assert any("2024-01-01" in t for t in texts)
    assert not [t for t in texts if "00:00" in t or "T00" in t]


# --- waffle shares when the sum overflows ---------------------------------------------------------------------------


def test_waffle_percent_survives_an_overflowing_sum():
    data = pd.DataFrame({"c": ["a", "b", "c"], "v": [1e308, 1e308, 0.5e308]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "overflow encountered" RuntimeWarning either
        res = vc.waffle(data, "c", "v", rows=4, columns=10)
    assert res.table["percent"].tolist() == pytest.approx([40.0, 40.0, 20.0])
    assert res.table["tiles"].tolist() == [16, 16, 8]


def test_waffle_percent_is_unchanged_for_ordinary_values():
    rng = np.random.default_rng(1)
    for _ in range(50):
        v = rng.random(6) * 10.0 ** rng.integers(-5, 8)
        res = vc.waffle(pd.DataFrame({"c": list("abcdef"), "v": v}), "c", "v", order=list("abcdef"))
        assert res.table["percent"].tolist() == (pd.Series(v) / pd.Series(v).sum() * 100).tolist()
        plt.close("all")


# --- percent_grid level, bullet bands, stacked_percentages columns --------------------------------------------------


@pytest.mark.parametrize("level", [95, 0, 1, -0.5, "0.95"])
def test_percent_grid_checks_level_before_drawing(level):
    with pytest.raises(ValueError, match="level must be strictly between 0 and 1"):
        vc.percent_grid(datasets.trial(), column="improved", facet="arm", level=level)
    assert plt.get_fignums() == []


@pytest.mark.parametrize(("bands", "message"), [
    (["poor", 300], r"bands must be a list of column names or a list of numbers, not a mix of both; "
                    r"got \['poor', 300\]"),
    ([100, np.nan], "bands must be a list of column names or of finite numbers"),
    ([100, None], "bands must be a list of column names or of finite numbers"),
    ([100, np.inf], "bands must be a list of column names or of finite numbers"),
])
def test_bullet_bands_are_validated(bands, message):
    with pytest.raises(ValueError, match=message):
        vc.bullet(pd.DataFrame({"k": ["x"], "v": [5], "poor": [3]}), "k", "v", bands=bands)
    assert plt.get_fignums() == []


def test_bullet_numeric_bands_in_any_sequence():
    data = pd.DataFrame({"k": ["x", "y"], "v": [5, 7]})
    for bands in ([3, 6.5], (np.int64(3), np.float64(6.5)), np.array([3, 6.5]), pd.Series([3, 6.5])):
        res = vc.bullet(data, "k", "v", bands=bands)
        assert [len(ax.patches) for ax in res.axes] == [3, 3]  # two bands and the measure
        plt.close("all")


@pytest.mark.parametrize(("roles", "call"), [
    ("group and category", lambda d: vc.stacked_percentages(d, group="m", category="m")),
    ("outer and inner", lambda d: vc.nested_pie(d, outer="m", inner="m")),
    ("stage and value", lambda d: vc.funnel(d, stage="v", value="v")),
])
def test_one_column_in_two_roles_is_rejected_clearly(roles, call):
    column = "v" if roles.startswith("stage") else "m"
    with pytest.raises(ValueError, match=f"{roles} must be different columns; both are '{column}'"):
        call(MONTHS)
    assert plt.get_fignums() == []


def test_circular_bar_column_may_be_both_label_and_group_or_value():
    data = pd.DataFrame({"city": ["x", "y", "z"], "v": [3.0, 1.0, 2.0]})
    res = vc.circular_bar(data, label="city", value="v", group="city")
    assert res.table["city"].tolist() == ["x", "y", "z"] and list(res.table.columns) == ["city", "v"]
    res = vc.circular_bar(data, label="v", value="v")
    assert res.table["v"].tolist() == [3.0, 2.0, 1.0]


def test_deprecated_category_order_alias_warns_at_the_callers_line():
    with pytest.warns(FutureWarning, match="category_order_") as record:
        vc.stacked_percentages(datasets.trial(), group="site", category="arm",
                               category_order_=["control", "low dose", "high dose"])
    assert record[0].filename == __file__
