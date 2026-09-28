"""Shared validation, thresholds, colours, labels and saving (compare, distribution, multivariate, explore, _core)."""

import inspect
from fractions import Fraction
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import viz_calc as vc
from viz_calc import compare, datasets, distribution, explore, multivariate
from viz_calc._core import abbreviate, category_order, cleanup_on_error, exact_mean, level_label


@pytest.fixture(scope="module")
def trial():
    return datasets.trial()


# --- threshold="mean" is correctly rounded, and ties sit on it ----------------------------------------------------


def test_exact_mean_is_correctly_rounded():
    rng = np.random.default_rng(3)
    for _ in range(300):
        n = int(rng.integers(1, 30))
        v = rng.normal(size=n) * 10.0 ** rng.integers(-6, 7, size=n)
        assert exact_mean(v) == float(sum(map(Fraction, v)) / n)
    assert exact_mean([4.2, 3.8, 3.4]) == 3.8  # math.fsum(...) / 3 gives 3.8000000000000003
    assert np.isnan(exact_mean([])) and np.isnan(exact_mean([1.0, np.nan]))
    assert exact_mean([1.0, np.inf]) == np.inf and np.isnan(exact_mean([np.inf, -np.inf]))
    assert exact_mean([1e308, 1e308, -1e308]) == 1e308 / 3  # the sum overflows, the mean does not
    assert exact_mean([1.5e308, 1.5e308]) == 1.5e308


def test_centered_bar_counts_a_value_on_the_mean_as_at_or_above():
    df = pd.DataFrame({"g": ["a"] * 3, "y": [4.2, 3.8, 3.4]})
    by_mean = vc.centered_bar(df, "g", "y", threshold="mean")
    by_number = vc.centered_bar(df, "g", "y", threshold=3.8)
    assert by_mean.info["threshold"] == 3.8
    assert by_mean.table["n_above"].tolist() == by_number.table["n_above"].tolist() == [2]


def test_centered_bar_mean_ties_with_values_of_a_larger_magnitude():
    # the mean (0.2) is small next to the values, so their rounding error is large relative to it
    df = pd.DataFrame({"g": ["a"] * 3, "y": [100.1, -99.7, 0.2]})
    assert vc.centered_bar(df, "g", "y").table["n_above"].tolist() == [2]


def test_benchmark_bar_groups_on_the_overall_mean_are_not_all_below():
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "y": [3.5, 4.3, 5.0, 2.8]})
    res = vc.benchmark_bar(df, "g", "y", threshold="mean", classify="mean")
    assert res.info["threshold"] == 3.9
    assert res.table["status"].tolist() == ["above", "above"]  # a mean on the threshold counts as above
    assert vc.benchmark_bar(df, "g", "y", threshold=3.9, classify="mean").table["status"].tolist() == ["above"] * 2


def test_benchmark_bar_constant_group_on_the_mean_is_indistinguishable():
    df = pd.DataFrame({"g": list("aabbcc"), "y": [4.2, 3.4, 3.8, 3.8, 3.0, 4.6]})
    table = vc.benchmark_bar(df, "g", "y").table.set_index("g")
    assert table.loc["b", "ci_low"] == table.loc["b", "ci_high"]
    assert table.loc["b", "status"] == "indistinguishable"
    # np.mean([3.8] * 3) is 3.7999999999999994: still on a threshold of 3.8
    df3 = pd.DataFrame({"g": list("aaabbb"), "y": [3.8] * 3 + [2.0, 3.8, 5.6]})
    assert vc.benchmark_bar(df3, "g", "y", threshold=3.8).table["status"].tolist() == ["indistinguishable"] * 2


def test_threshold_still_separates_real_differences():
    df = pd.DataFrame({"g": list("aabb"), "y": [1.0, 1.0, 1.0 + 1e-9, 1.0 + 1e-9]})
    assert vc.centered_bar(df, "g", "y", threshold=1.0 + 1e-9).table["n_above"].tolist() == [0, 2]
    t = vc.benchmark_bar(df, "g", "y", threshold=1.0 + 5e-10, classify="mean").table
    assert t["status"].tolist() == ["below", "above"]


@pytest.mark.parametrize("func", [vc.benchmark_bar, vc.centered_bar])
@pytest.mark.parametrize("threshold", ["Mean", "avg", True, np.nan, None])
def test_bad_threshold_raises_the_same_clear_error(trial, func, threshold):
    with pytest.raises(ValueError, match="threshold must be a finite number, 'mean' or 'median'"):
        func(trial, x="arm", y="score", threshold=threshold)


# --- nullable dtypes (pandas 2.0) ---------------------------------------------------------------------------------


@pytest.mark.parametrize("missing", [False, True])
def test_profile_scatters_accepts_nullable_numeric_columns(missing):
    m = datasets.measurements()
    if missing:
        m.loc[m.index[::9], ["length", "mass"]] = np.nan
    plain = vc.profile_scatters(m)
    nullable = vc.profile_scatters(m.convert_dtypes())
    pd.testing.assert_frame_equal(plain.table, nullable.table)


@pytest.mark.parametrize("values,expected", [
    ([True, False, True], {"False": 1, "True": 2}),
    ([True, None, True], {"True": 2, "(missing)": 1}),
])
def test_profile_bars_shows_missing_only_when_something_is_missing(values, expected):
    df = pd.DataFrame({"flag": pd.array(values, dtype="boolean")})
    table = vc.profile_bars(df).table
    assert dict(zip(table["level"], table["count"])) == expected


# --- radar needs a polar Axes -------------------------------------------------------------------------------------


def test_radar_rejects_a_cartesian_axes_before_drawing():
    m = datasets.measurements()
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="polar Axes"):
        vc.radar(m, metrics=["length", "width", "depth", "mass"], group="group", ax=ax)
    assert not ax.lines and not ax.patches and not ax.texts
    fig = plt.figure()
    polar = fig.add_subplot(1, 2, 2, projection="polar")
    assert vc.radar(m, metrics=["length", "width", "depth", "mass"], group="group", ax=polar).axes is polar


# --- colours ------------------------------------------------------------------------------------------------------


def _fixed_slot_calls(trial):
    wide = trial.head(6).assign(item=list("abcdef"), pos=lambda d: d["score"].abs(), base=lambda d: d["baseline"].abs())
    return {
        "benchmark_bar": (3, lambda c: vc.benchmark_bar(trial, x="arm", y="score", colors=c)),
        "centered_bar": (2, lambda c: vc.centered_bar(trial, x="arm", y="score", colors=c)),
        "dumbbell": (2, lambda c: vc.dumbbell(wide, label="item", start="baseline", end="score", colors=c)),
        "divergent_bar": (2, lambda c: vc.divergent_bar(wide, category="item", left="base", right="pos", colors=c)),
    }


@pytest.mark.parametrize("name", ["benchmark_bar", "centered_bar", "dumbbell", "divergent_bar"])
def test_fixed_slot_colors_are_validated(trial, name):
    k, call = _fixed_slot_calls(trial)[name]
    for bad in ["viridis", "red", ("red",) * (k + 1), ("red",) * (k - 1)]:
        with pytest.raises(ValueError, match=f"colors must be a tuple of {k} colours"):
            call(bad)
    with pytest.raises(ValueError, match="invalid colour"):
        call(("red",) * (k - 1) + ("not-a-colour",))
    assert plt.get_fignums() == []  # nothing was drawn before the check
    call(["red"] * k)


def test_benchmark_bar_honours_its_colours(trial):
    res = vc.benchmark_bar(trial, x="arm", y="score", colors=["#111111", "#222222", "#333333"])
    colour_of = {"below": "#111111", "above": "#222222", "indistinguishable": "#333333"}
    assert [to_hex(p.get_facecolor()) for p in res.axes.patches] == [colour_of[s] for s in res.table["status"]]


def test_likert_colors_need_one_per_level():
    s = datasets.survey()
    items = ["Workload is manageable", "Tools are adequate"]
    levels = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
    with pytest.raises(ValueError, match="5 colours"):
        vc.likert(s, items=items, levels=levels, colors=["red", "blue"])
    vc.likert(s, items=items, levels=levels, colors=["red", "orange", "grey", "lightblue", "blue"])
    vc.likert(s, items=items, levels=levels, colors="PuOr")


def test_palette_rejects_invalid_colours():
    with pytest.raises(ValueError, match="invalid colour"):
        vc.palette(3, ["red", "nope"])
    assert vc.palette(3, [(0.1, 0.2, 0.3)]) == [(0.1, 0.2, 0.3)] * 3


# --- option validation --------------------------------------------------------------------------------------------


def test_lollipop_sort_is_validated(trial):
    with pytest.raises(ValueError, match=r"sort must be one of \['descending', 'ascending', None\]"):
        vc.lollipop(trial, x="arm", y="score", sort="asc")
    means = trial.groupby("arm", observed=True)["score"].mean()
    assert vc.lollipop(trial, x="arm", y="score", sort="ascending").table["value"].is_monotonic_increasing
    assert vc.lollipop(trial, x="arm", y="score", sort=None).table["arm"].tolist() == means.index.tolist()


@pytest.mark.parametrize("kwargs,match", [
    ({"ref_line": "avg"}, r"ref_line must be one of \['mean', 'median'\]"),
    ({"ref_line": True}, "ref_line must be 'mean', 'median', a number or None"),
    ({"ref_line": [1, 2]}, "ref_line must be 'mean', 'median', a number or None"),
    ({"bins": "freedman"}, "bins must be one of"),
    ({"facet": "site", "col_wrap": 0}, "col_wrap must be a whole number of at least 1"),
    ({"facet": "site", "col_wrap": 2.0}, "col_wrap must be a whole number of at least 1"),
])
def test_histogram_options_are_validated_before_drawing(trial, kwargs, match):
    with pytest.raises(ValueError, match=match):
        vc.histogram(trial, x="score", **kwargs)
    assert plt.get_fignums() == []


def test_histogram_numeric_ref_line(trial):
    res = vc.histogram(trial, x="score", ref_line=0)
    assert [line.get_xdata()[0] for line in res.axes[0, 0].lines] == [0.0]


def test_estimation_plot_p_adjust_is_checked_before_drawing(trial):
    with pytest.raises(ValueError, match=r"p_adjust must be one of \['holm', 'fdr_bh', 'bonferroni', 'none'\]"):
        vc.estimation_plot(trial, x="arm", y="score", p_adjust="BH", n_resamples=100)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("func,kwargs,match", [
    (vc.profile_bars, {"per_page": 0}, "per_page"),
    (vc.profile_bars, {"ncols": 0}, "ncols"),
    (vc.profile_boxes, {"per_page": 0}, "per_page"),
    (vc.profile_boxes, {"ncols": -1}, "ncols"),
    (vc.profile_scatters, {"ncols": 0}, "ncols"),
    (vc.profile_scatters, {"max_points": 0}, "max_points"),
    (vc.profile_scatters, {"p_adjust": "BH"}, "p_adjust must be one of"),
    (vc.profile_scatters, {"method": "kendall"}, "method must be one of"),
])
def test_profile_options_are_validated(func, kwargs, match):
    with pytest.raises(ValueError, match=match):
        func(datasets.measurements(), **kwargs)
    assert plt.get_fignums() == []


# --- columns with no data -----------------------------------------------------------------------------------------


def _blank(df, column):
    df = df.copy()
    df[column] = np.nan
    return df


@pytest.mark.parametrize("column,call", [
    ("arm", lambda d: vc.lollipop(d, x="arm", y="score")),
    ("arm", lambda d: vc.lollipop(d, x="arm", stat="count")),
    ("score", lambda d: vc.lollipop(d, x="arm", y="score")),
    ("arm", lambda d: vc.divergent_bar(d.assign(p=d["score"].abs()), category="arm", left="baseline", right="p")),
    ("score", lambda d: vc.ridgeplot(d, x="score", group="arm")),
    ("score", lambda d: vc.benchmark_bar(d, x="arm", y="score")),
    ("score", lambda d: vc.benchmark_bar(d, x="arm", y="score", threshold=0)),
    ("score", lambda d: vc.centered_bar(d, x="arm", y="score", threshold="median")),
    ("score", lambda d: vc.raincloud(d, x="arm", y="score")),
    ("score", lambda d: vc.histogram(d, x="score", bins=10)),
])
def test_all_missing_columns_raise_a_clear_error(trial, column, call):
    with pytest.raises(ValueError, match=f"column '{column}' has no non-missing values"):
        call(_blank(trial, column))
    assert plt.get_fignums() == []


# --- column lists ---------------------------------------------------------------------------------------------------


def test_profile_bars_accepts_an_index_or_array_of_columns():
    s = datasets.survey()
    expected = vc.profile_bars(s, columns=["team", "remote"]).table
    pd.testing.assert_frame_equal(vc.profile_bars(s, columns=pd.Index(["team", "remote"])).table, expected)
    pd.testing.assert_frame_equal(vc.profile_bars(s, columns=np.array(["team", "remote"])).table, expected)
    m = datasets.measurements()
    vc.profile_boxes(m, categorical=pd.Index(["group"]), numeric=m.select_dtypes("number").columns)
    vc.pca_plot(m, features=m.select_dtypes("number").columns, group="group")


@pytest.mark.parametrize("call,name", [
    (lambda d: vc.profile_scatters(d, columns="xy"), "columns"),
    (lambda d: vc.profile_bars(d, columns="xy"), "columns"),
    (lambda d: vc.profile_boxes(d, numeric="xy"), "numeric"),
    (lambda d: vc.pca(d, features="xy"), "features"),
    (lambda d: vc.pca_plot(d, features="xy"), "features"),
    (lambda d: vc.radar(d, metrics="xyz", group="g"), "metrics"),
    (lambda d: vc.likert(d, items="xy", levels=[1, 2]), "items"),
])
def test_a_string_is_not_split_into_single_letter_columns(call, name):
    df = pd.DataFrame({"x": np.arange(12.0), "y": np.arange(12.0) ** 2, "z": np.arange(12.0) % 5,
                       "xy": np.arange(12.0)[::-1], "xyz": np.arange(12.0), "g": list("ab") * 6})
    with pytest.raises(TypeError, match=f"{name} must be a list of column names, not a str"):
        call(df)


def test_order_and_levels_must_not_be_a_string(trial):
    with pytest.raises(TypeError, match="order for 'arm' must be a list of levels"):
        vc.benchmark_bar(trial, x="arm", y="score", order="control")
    with pytest.raises(TypeError, match="levels must be a list"):
        vc.likert(datasets.survey(), items=["remote"], levels="yn")


# --- datetime levels ------------------------------------------------------------------------------------------------


@pytest.fixture
def monthly():
    return pd.DataFrame({"month": pd.to_datetime(["2024-01-01", "2024-02-01"] * 4), "y": np.arange(8.0)})


def test_category_order_matches_datetime_orders_by_value(monthly):
    values = monthly["month"]
    stamps = list(values.drop_duplicates())[::-1]
    raw = list(pd.unique(values))[::-1]  # numpy.datetime64: hashes differently from Timestamp on NumPy 1.x
    for order in (stamps, raw):
        assert category_order(values, order, complete=True) == order
        res = vc.benchmark_bar(monthly, x="month", y="y", order=order)
        assert [t.get_text() for t in res.axes.get_xticklabels()] == ["2024-02-01", "2024-01-01"]
    with pytest.raises(ValueError, match="missing"):
        category_order(values, stamps[:1], complete=True)
    levels = category_order(values)
    assert levels == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]
    assert all(isinstance(v, pd.Timestamp) for v in levels)


def test_datetime_levels_are_labelled_as_dates(monthly):
    assert [level_label(v) for v in (pd.Timestamp("2024-01-01"), np.datetime64("2024-01-01T00:00:00.000000000"))] \
        == ["2024-01-01", "2024-01-01"]
    assert level_label(pd.Timestamp("2024-01-01 06:30")) == "2024-01-01 06:30:00"
    assert level_label(pd.Timestamp("2024-01-01", tz="UTC")) == str(pd.Timestamp("2024-01-01", tz="UTC"))
    assert level_label(np.timedelta64(1, "D")) == str(pd.Timedelta(days=1))
    assert level_label("a") == "a" and level_label(3) == "3"
    res = vc.histogram(monthly, x="y", facet="month", hue="month")
    assert [a.get_title(loc="left") for a in res.axes.flat] == ["month = 2024-01-01", "month = 2024-02-01"]
    assert [t.get_text() for t in vc.lollipop(monthly, x="month", y="y").axes.get_yticklabels()] \
        == ["2024-02-01", "2024-01-01"]


# --- abbreviate -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("num,text", [
    (999_999, "1M"), (999_950, "1M"), (999_949, "999.9K"), (999_999_999, "1B"), (999_999_999_999, "1T"),
    (-999_999, "-1M"), (9_999.96, "10K"), (9_999.94, "9,999.9"), (10_000, "10K"), (12_345, "12.3K"),
    (1_000_000, "1M"), (2.5e15, "2500T"), (0.99999, "1"),
])
def test_abbreviate_chooses_the_unit_after_rounding(num, text):
    assert abbreviate(num) == text


def test_abbreviate_with_zero_digits():
    assert abbreviate(999_600, digits=0) == "1M"
    assert abbreviate(9_999.6, digits=0) == "10K"


# --- saving multi-page results --------------------------------------------------------------------------------------


def test_multi_page_result_saves_to_a_pdf_stream():
    res = vc.profile_boxes(datasets.measurements(), per_page=1)
    assert len(res.figure) > 1
    buf = BytesIO()
    assert res.save(buf, format="pdf") is buf
    data = buf.getvalue()
    assert data.startswith(b"%PDF")
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") == len(res.figure)
    with pytest.raises(ValueError, match="needs a file path"):
        res.save(BytesIO())


def test_multi_page_result_still_saves_numbered_files(tmp_path):
    res = vc.profile_boxes(datasets.measurements(), per_page=2)
    paths = res.save(tmp_path / "boxes.png")
    assert [p.split("/")[-1] for p in paths] == [f"boxes_{i}.png" for i in range(1, len(res.figure) + 1)]


# --- figures are closed when a call fails ---------------------------------------------------------------------------


def test_cleanup_on_error_closes_only_the_figures_the_call_opened():
    keep, _ = plt.subplots()

    @cleanup_on_error
    def fails(n):
        """Doc."""
        for _ in range(n):
            plt.subplots()
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        fails(2)
    assert plt.get_fignums() == [keep.number]
    assert fails.__name__ == "fails" and fails.__doc__ == "Doc." and list(inspect.signature(fails).parameters) == ["n"]


def test_a_call_that_fails_after_drawing_leaves_no_figure(trial, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("welch failed")

    monkeypatch.setattr(compare.st, "welch_test", broken)
    with pytest.raises(RuntimeError, match="welch failed"):
        vc.estimation_plot(trial, x="arm", y="score", n_resamples=100)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("module", [compare, distribution, multivariate, explore])
def test_public_plotting_functions_clean_up_on_error(module):
    for name in module.__all__:
        func = getattr(module, name)
        if name in ("pca", "to_pptx"):  # they draw nothing
            continue
        assert hasattr(func, "__wrapped__"), name
        assert inspect.signature(func) == inspect.signature(func.__wrapped__)
