"""The documentation matches the code.

Every Python block on the pages below is run in page order, and each results
table shown after a block is compared with what that block computes.
"""

import functools
import inspect
import os
import re
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets, stats

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["README.md", "docs/index.md", "docs/getting-started.md",
         "docs/walkthroughs/comparing-groups.md", "docs/walkthroughs/correlations.md",
         "docs/walkthroughs/proportions-and-surveys.md", "docs/walkthroughs/exploring-a-dataset.md",
         "docs/walkthroughs/multivariate-and-networks.md", "docs/walkthroughs/time-and-projects.md"]
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
NOT_CHARTS = {"palette", "pca", "to_pptx", "VizResult"}
CHARTS = sorted(n for n in vc.__all__ if callable(getattr(vc, n)) and n not in NOT_CHARTS)


def _read(page: str) -> str:
    return (ROOT / page).read_text(encoding="utf-8")


@functools.cache
def _run(page: str) -> list[dict]:
    """Run the page's Python blocks in order; record each block's span, error and ``res``."""
    text, ns, records = _read(page), {}, []
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:  # blocks save files
        os.chdir(tmp)
        try:
            for m in BLOCK.finditer(text):
                rec = {"start": m.start(), "end": m.end(), "code": m.group(1), "error": None}
                try:
                    exec(compile(m.group(1), page, "exec"), ns)
                except Exception as exc:  # reported by test_code_blocks_run
                    rec["error"] = exc
                res = ns.get("res")
                if hasattr(res, "table"):
                    rec["table"] = None if res.table is None else res.table.copy()
                    rec["comparisons"] = (res.info or {}).get("comparisons")
                records.append(rec)
                plt.close("all")
                if rec["error"] is not None:
                    break
        finally:
            os.chdir(here)
    return records


@pytest.mark.parametrize("page", PAGES)
def test_code_blocks_run(page):
    records = _run(page)
    assert records, f"{page} has no Python blocks"
    err = records[-1]["error"]
    if isinstance(err, ImportError) and "optional dependency" in str(err):
        pytest.skip(f"{page}: {err}")
    assert err is None, f"{page}: block\n{records[-1]['code']}\nraised {err!r}"


# --- tables shown after a block ------------------------------------------------------------------------------------

ALIASES = {"ρ": "r", "%": "percent", "η²": "eta_squared", "A": "group_a", "B": "group_b", "r(A)": "r_a",
           "r(B)": "r_b", "Δr": "difference", "adjusted p": "p_adjusted", "Welch p": "p",
           "Holm-adjusted p": "p_adjusted"}
SPLIT = {"95% CI": ["ci_low", "ci_high"], "Hedges' g [95% CI]": ["hedges_g", "g_ci_low", "g_ci_high"],
         "pair": ["var1", "var2"]}


def _md_table(text: str, start: int, stop: int) -> list[dict[str, str]]:
    """The first Markdown table in ``text[start:stop]`` as a list of {column: cell}, aliases resolved."""
    lines = [ln.strip() for ln in text[start:stop].splitlines()]
    first = next((i for i, ln in enumerate(lines) if ln.startswith("|")), None)
    if first is None:
        return []
    rows = []
    for ln in lines[first:]:
        if not ln.startswith("|"):
            break
        rows.append([c.strip() for c in ln.strip("|").split("|")])
    header, body = rows[0], rows[2:]
    out = []
    for cells in body:
        row = {}
        for h, c in zip(header, cells):
            if h in SPLIT:
                for col, part in zip(SPLIT[h], re.split(r" to |–|, | \[|\]", c)):
                    row[col] = part
            else:
                row[ALIASES.get(h, h)] = c
        out.append(row)
    return out


def _number(cell: str) -> tuple[float, int] | None:
    """``"−0.36"`` -> (-0.36, 2); ``"+3.3"`` -> (3.3, 1); None if not a number."""
    s = cell.replace("−", "-").replace("+", "").replace(",", "").strip()
    try:
        value = float(s)
    except ValueError:
        return None
    return value, len(s.split(".")[1]) if "." in s else 0


def _assert_row(doc: dict[str, str], actual: pd.Series, where: str) -> None:
    for col, cell in doc.items():
        if col not in actual.index or not cell:
            continue
        got = actual[col]
        if cell.startswith("<"):  # "< .001"
            assert got < float(cell.lstrip("< ")), f"{where}: {col} = {got}, docs say {cell}"
        elif (num := _number(cell)) is not None and not isinstance(got, str):
            value, decimals = num
            assert abs(float(got) - value) <= 0.5 * 10.0**-decimals + 1e-9, f"{where}: {col} = {got}, docs say {cell}"
        else:
            assert str(got) == cell, f"{where}: {col} = {got!r}, docs say {cell!r}"


TABLES = [  # page, text in the block that makes the table, where the numbers are, key columns, rows in table order?
    ("docs/getting-started.md", "vc.benchmark_bar(", "table", ["arm"], True),
    ("docs/walkthroughs/comparing-groups.md", "vc.raincloud(", "table", ["arm"], True),
    ("docs/walkthroughs/comparing-groups.md", "vc.estimation_plot(", "comparisons", ["group"], True),
    ("docs/walkthroughs/correlations.md", "vc.correlogram(", "table", ["var1", "var2"], True),
    ("docs/walkthroughs/correlations.md", "vc.compare_correlations(", "table", ["group_a", "group_b", "var1", "var2"],
     False),
    ("docs/walkthroughs/proportions-and-surveys.md", "vc.percent_grid(", "table", ["facet"], True),
    ("docs/walkthroughs/proportions-and-surveys.md", "vc.likert(", "table", ["item"], True),
    ("docs/walkthroughs/exploring-a-dataset.md", "vc.profile_boxes(", "table", ["categorical", "numeric"], True),
    ("docs/walkthroughs/exploring-a-dataset.md", "vc.profile_scatters(", "table", ["x", "y"], True),
    ("docs/walkthroughs/multivariate-and-networks.md", "vc.network_map(", "table", ["node"], False),
    ("docs/walkthroughs/multivariate-and-networks.md", "vc.sankey(", "table", ["node"], True),
]


@pytest.mark.parametrize("page,call,source,keys,ordered", TABLES, ids=[f"{p.split('/')[-1]}:{c}" for p, c, *_ in TABLES])
def test_documented_table_matches_code(page, call, source, keys, ordered):
    text, records = _read(page), _run(page)
    i = next(i for i, r in enumerate(records) if call in r["code"])
    rec = records[i]
    if isinstance(rec["error"], ImportError):
        pytest.skip(str(rec["error"]))
    assert rec["error"] is None, rec["error"]
    stop = records[i + 1]["start"] if i + 1 < len(records) else len(text)
    doc_rows = [r for r in _md_table(text, rec["end"], stop) if not any("…" in c or "*" in c for c in r.values())]
    assert doc_rows, f"{page}: no results table after the block with {call}"
    actual = rec[source]
    missing = [k for k in keys if k not in doc_rows[0]]
    assert not missing, f"{page}: table after {call} lacks key column(s) {missing}; res.{source} has {list(actual)}"
    unknown = [c for c in doc_rows[0] if c not in actual.columns]
    assert not unknown, f"{page}: table after {call} has column(s) {unknown} that res.{source} does not have"
    index = actual.assign(_key=actual[keys].astype(str).agg("|".join, axis=1)).set_index("_key")
    doc_keys = ["|".join(r[k] for k in keys) for r in doc_rows]
    for key, row in zip(doc_keys, doc_rows):
        assert key in index.index, f"{page}: row {key!r} is not in res.{source}"
        _assert_row(row, index.loc[key], f"{page} [{key}]")
    if ordered:
        assert doc_keys == list(index.index[: len(doc_keys)]), f"{page}: rows are not in the order the code gives"


def test_correlogram_spearman_ci_and_percent_grid_facet_column():
    # the two cells that used to disagree with the code
    meas, trial = datasets.measurements(), datasets.trial()
    t = vc.correlogram(meas, method="spearman").table.set_index(["var1", "var2"])
    assert f"{t.loc[('length', 'depth'), 'ci_low']:.2f}" == "0.84"
    assert "| length | depth | 0.89 | 0.84 to 0.92 | 150 |" in _read("docs/walkthroughs/correlations.md")
    assert "facet" in vc.percent_grid(trial, column="improved", facet="arm").table.columns
    assert "| facet | n | successes | percent | ci_low | ci_high |" in _read("docs/walkthroughs/proportions-and-surveys.md")


def test_compare_correlations_every_pair_row():
    # "| beta | gamma | *every pair* | | | within ±0.23 | | 1.00 |"
    t = vc.compare_correlations(datasets.measurements(), group="group").table
    bg = t[(t.group_a == "beta") & (t.group_b == "gamma")]
    assert len(bg) == 6
    assert bg["difference"].abs().max() < 0.235
    assert (bg["p_adjusted"].round(2) == 1.0).all()


def test_getting_started_statistics_comments():
    text = _read("docs/getting-started.md")
    trial = datasets.trial()
    w = stats.welch_test(trial.loc[trial["arm"] == "control", "score"], trial.loc[trial["arm"] == "high dose", "score"])
    shown = f"treated − control: {w['difference']:.1f}, 95% CI {w['ci_low']:.1f} to {w['ci_high']:.1f}, p = {w['p']:.3f}"
    assert shown in text
    lo, hi = stats.wilson_ci(12, 40)
    assert f"({lo:.3f}, {hi:.3f})" in text
    assert list(stats.adjust_pvalues([0.01, 0.04, 0.03], "holm").round(2)) == [0.03, 0.06, 0.06]
    assert "[0.03, 0.06, 0.06]" in text


# --- colours and ax= ------------------------------------------------------------------------------------------------


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end < 0 else text[start:end]


def _params(name: str) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(getattr(vc, name)).parameters)


def test_colour_table_matches_signatures():
    rows = [ln.strip("|").split("|") for ln in _section(_read("docs/getting-started.md"), "## Colours").splitlines()
            if ln.startswith("| ")]
    listed = {}
    for arg, _takes, funcs in rows[1:]:
        for name, roles in re.findall(r"`(\w+)`(?: \(([^)]*)\))?", funcs):
            if name in CHARTS:
                assert name not in listed, f"{name} is listed twice"
                listed[name] = (arg.strip(), roles)
    assert sorted(listed) == CHARTS, f"not in the colour table: {sorted(set(CHARTS) - set(listed))}"
    for name, (arg, roles) in listed.items():
        params = _params(name)
        colour_args = {p for p in ("colors", "color", "cmap", "bar_color") if p in params}
        if arg == "none":
            assert not colour_args, f"{name} is listed without a colour argument but has {colour_args}"
            continue
        param = arg.strip("`=")
        assert param in params, f"{name} has no {param}="
        if param == "colors" and roles:  # a fixed tuple, one colour per role
            default = params["colors"].default
            assert isinstance(default, tuple) and len(default) == len(roles.split(", ")), (name, default, roles)
        elif param == "colors":  # a colormap name or a list, through palette()
            assert not isinstance(params["colors"].default, tuple), name


def test_colormap_names_work_where_the_docs_say_so():
    trial, meas = datasets.trial(), datasets.measurements()
    vc.raincloud(trial, x="arm", y="score", colors="viridis")
    vc.radar(meas, metrics=["length", "width", "depth", "mass"], group="group", colors=["#000000", "#555555"])
    vc.likert(datasets.survey(), items=["Tools are adequate"],
              levels=["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"], colors="PuOr")


def test_ax_claims_match_signatures():
    text = _read("docs/getting-started.md")
    section = _section(text, "## Putting charts into your own layouts")
    own_figure = set()
    for ln in section.splitlines():
        if ln.startswith("| `"):
            own_figure |= {n for n in re.findall(r"`(\w+)`", ln.split("|")[1]) if n in CHARTS}
    assert own_figure, "no list of functions that create their own figure"
    for name in CHARTS:
        if name in own_figure:
            assert "ax" not in _params(name), f"{name} is listed as creating its own figure but takes ax="
        else:
            assert "ax" in _params(name), f"{name} has no ax= but is not listed as creating its own figure"


def test_conventions_note_matches_signatures():
    section = _section(_read("docs/getting-started.md"), "## Conventions")
    for token in re.findall(r"`([^`]+)`", section):  # every name is a chart function or one of their arguments
        name = token.split("=")[0]
        if name not in CHARTS and name not in ("None", "profile_*"):
            assert any(name in _params(f) for f in CHARTS), f"`{token}` in Conventions is not an argument"

    def ann(func, param):
        return str(_params(func)[param].annotation)

    for f in ["correlogram", "compare_correlations", "donut_grid", "profile_bars", "profile_scatters"]:
        assert ann(f, "columns").startswith("Sequence[str]"), f
    assert ann("waffle", "columns") == "int" and ann("waffle", "rows") == "int"
    for f in ["lollipop", "centered_bar", "likert", "donut_grid", "nested_pie", "network_map", "profile_bars"]:
        assert ann(f, "labels") == "bool", f
    assert ann("timeseries_fill", "labels").startswith("Sequence[str]")
    assert _params("correlogram")["alpha"].default == _params("compare_correlations")["alpha"].default == 0.05
    assert "alpha" in _params("histogram") and "alpha" in _params("timeseries_fill")
    for f in ["estimation_plot", "benchmark_bar", "lollipop", "centered_bar", "raincloud", "histogram", "ridgeplot",
              "quadrant_plot", "animated_bubble"]:
        assert "x" in _params(f), f
    for f in ["ridgeplot", "compare_correlations", "stacked_percentages", "circular_bar", "gantt", "pca_plot", "radar"]:
        assert "group" in _params(f), f
    for f in ["histogram", "percent_grid", "donut_grid"]:
        assert "col_wrap" in _params(f), f
    for f in ["profile_bars", "profile_boxes", "profile_scatters"]:
        assert "ncols" in _params(f), f


def test_polar_functions_draw_into_a_polar_axes():
    meas = datasets.measurements()
    fig = plt.figure()
    spider = fig.add_subplot(1, 2, 2, projection="polar")
    assert vc.radar(meas, metrics=["length", "width", "depth", "mass"], group="group", ax=spider).axes is spider
    _, ring = plt.subplots(subplot_kw={"projection": "polar"})
    small = meas.head(5).assign(id=list("abcde"))
    assert vc.circular_bar(small, label="id", value="mass", ax=ring).axes is ring
