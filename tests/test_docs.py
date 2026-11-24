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
    i = next((i for i, r in enumerate(records) if call in r["code"]), None)
    err = records[-1]["error"]
    if i is None and isinstance(err, ImportError) and "optional dependency" in str(err):
        pytest.skip(f"{page} stopped before the block with {call}: {err}")
    assert i is not None, f"{page}: no block contains {call}"
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


def test_showing_figures_says_how_to_close_them():
    section = _section(_read("docs/getting-started.md"), "## Showing figures")
    open_before = set(plt.get_fignums())
    res = vc.waffle(datasets.survey(), category="team")
    assert set(plt.get_fignums()) > open_before  # pyplot keeps every chart's figure until it is closed
    assert "plt.close(res.figure)" in section
    plt.close(res.figure)
    pages = vc.profile_boxes(datasets.measurements(), per_page=6).figure
    assert isinstance(pages, list) and len(pages) == 2  # plt.close(pages) would raise TypeError
    assert "for page in res.figure: plt.close(page)" in section
    for page in pages:
        plt.close(page)
    assert set(plt.get_fignums()) == open_before


def test_getting_started_describes_info_as_the_charts_fill_it():
    pattern = _section(_read("docs/getting-started.md"), "## The one pattern to learn")
    comment = next(ln for ln in pattern.splitlines() if ln.startswith("res.info"))
    info = vc.benchmark_bar(datasets.trial(), x="arm", y="score", threshold=50).info
    assert "threshold" in info and "threshold" in comment  # the comment describes the call above it
    # not every chart records its settings: a chart with nothing beyond its table returns an empty dict
    df = pd.DataFrame({"m": ["a", "b"], "s": [1.0, 2.0], "o": [2.0, 1.0]})
    simple = [vc.dumbbell(df, label="m", start="s", end="o"), vc.funnel(df, stage="m", value="o"),
              vc.waterfall(df, label="m", value="s")]
    assert any(res.info == {} for res in simple)
    assert "empty dict" in " ".join(pattern.split())


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


def test_colours_section_names_each_default_colormap():
    # a chart whose default is a colormap rather than the Okabe-Ito palette says so next to its name
    section = _section(_read("docs/getting-started.md"), "## Colours")
    units = [ln for ln in section.splitlines() if ln.startswith("|")]
    units += [par for par in section.split("\n\n") if not par.startswith("|")]
    for name in CHARTS:
        for param in ("colors", "cmap"):
            default = _params(name)[param].default if param in _params(name) else None
            if isinstance(default, str):
                assert any(f"`{name}`" in u and f'`"{default}"`' in u and "default" in u for u in units), \
                    f"the Colours section does not say that {name}'s {param}= defaults to {default!r}"


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


# --- API reference -------------------------------------------------------------------------------------------------


def _api_objects() -> dict[str, object]:
    """{dotted name: object} for each entry of docs/api.md: the top-level names and the stats and datasets functions."""
    objs = {f"viz_calc.{n}": getattr(vc, n) for n in vc.__all__ if n not in ("stats", "datasets")}
    for module in (stats, datasets):
        objs.update({f"{module.__name__}.{n}": f for n, f in vars(module).items()
                     if not n.startswith("_") and inspect.isfunction(f) and f.__module__ == module.__name__})
    return objs


def test_api_docstrings_use_links_the_api_page_renders():
    # mkdocstrings reads Markdown: a Sphinx role such as :func:`gantt` shows up as the literal text ":func:gantt"
    objs = _api_objects()
    for name, obj in objs.items():
        doc = inspect.getdoc(obj) or ""
        assert not re.search(r":\w+:`", doc), f"{name} has a Sphinx role: {re.search(r':[a-z]+:`[^`]*`', doc)[0]}"
        for target in re.findall(r"\]\[([\w.]+)\]", doc):  # [`text`][viz_calc.name] links to another API entry
            assert target in objs, f"{name} links to {target}, which the API page does not render"


def test_api_page_says_where_each_entry_imports_from():
    text = _read("docs/api.md")
    intro = text.split("\n## ")[0]
    for name in re.findall(r"^::: viz_calc\.(\w+)$", text, re.M):
        if name in ("stats", "datasets"):  # their functions are not top-level names (vc.wilson_ci does not exist)
            assert f"`viz_calc.{name}`" in intro, f"docs/api.md does not say the {name} functions are in viz_calc.{name}"
        else:
            assert name in vc.__all__, f"docs/api.md says {name} imports from the top level"


def test_okabe_ito_has_a_docstring_the_api_page_shows():
    # mkdocstrings shows a module constant's description only from a string right after the assignment
    import ast

    body = ast.parse(Path(inspect.getsourcefile(vc.palette)).read_text(encoding="utf-8")).body
    i = next(i for i, node in enumerate(body) if isinstance(node, ast.AnnAssign | ast.Assign)
             and "OKABE_ITO" in ast.unparse(node.target if isinstance(node, ast.AnnAssign) else node.targets[0]))
    doc = body[i + 1]
    assert isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant) and isinstance(doc.value.value, str)
    assert "Okabe" in doc.value.value and "colour-vision" in doc.value.value


def test_api_reference_lists_have_one_paragraph_per_entry():
    # the API page renders lines that are not separated by a blank line as one run-on paragraph
    citation = re.compile(r"^[A-Z][^\s,]*, [A-Z]\.")  # "Surname, X." starts an entry
    for name, obj in _api_objects().items():
        doc = inspect.getdoc(obj) or ""
        if "References\n----------\n" not in doc:
            continue
        section = re.split(r"\n\n\w+\n-+\n", doc.split("References\n----------\n")[1])[0].splitlines()
        assert citation.match(section[0]), f"{name}: References does not start with an entry"
        joined = [line for before, line in zip(section, section[1:]) if citation.match(line) and before]
        assert not joined, f"{name}: no blank line before the reference {joined[0]!r}"


def _doc(obj) -> str:
    """The docstring with its line breaks folded, so phrases can be searched across lines."""
    return " ".join((inspect.getdoc(obj) or "").split())


def _documented(func) -> list[str]:
    """The names in a function's numpy-style Parameters section."""
    section = re.split(r"\n\n\w+\n-+\n", inspect.getdoc(func).split("Parameters\n----------\n")[1])[0]
    return re.findall(r"^(\w+)$", section, re.M)


def test_docstrings_match_the_code():
    # functions that describe their parameters one by one leave none out
    for func in (vc.estimation_plot, vc.percent_grid):
        undocumented = set(inspect.signature(func).parameters) - set(_documented(func)) - {"data"}
        assert not undocumented, f"{func.__name__} does not document {sorted(undocumented)}"
    # donut_grid: labels=False removes the share labels and keeps the totals in the centre
    res = vc.donut_grid(pd.DataFrame({"a": [1, 3], "b": [3, 3]}), labels=False)
    assert [t.get_text() for ax in res.axes.flat for t in ax.texts if t.get_text()] == ["4", "6"]
    assert "labels=False`` leaves out every share but keeps the totals" in _doc(vc.donut_grid)
    # ridgeplot: a group with fewer than two distinct values gets no curve and so no median mark
    df = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 3 + ["c"], "y": [1, 2, 3, 4, 5, 7, 7, 7, 9]})
    marks = [line for line in vc.ridgeplot(df, x="y", group="g").axes.lines if line.get_color() == "black"]
    assert len(marks) == 1
    assert "fewer than two distinct values has no curve and no median mark" in _doc(vc.ridgeplot)


def test_mkdocs_edit_uri_only_with_page_actions():
    # Material uses edit_uri only for the edit and view buttons, which are off unless listed as theme features
    config = _read("mkdocs.yml")
    if re.search(r"^edit_uri:", config, re.M):
        assert re.search(r"^\s+- content\.action\.(edit|view)$", config, re.M), "edit_uri is set but has no effect"


# --- methodology and citations -------------------------------------------------------------------------------------

_NAME = r"[A-Z][\w\-]*"
# (first author, year) of an author-year citation: "Welch, 1947", "Delacre, Lakens & Leys, 2017", "Ho et al., 2019",
# "Scott (1979)", "Steiger's (1980)", "**Benjamini–Hochberg** (1995)"; names may wrap onto the next line
CITATION = re.compile(rf"\b({_NAME})(?:–{_NAME})?(?:\s+et\s+al\.|(?:,\s+{_NAME})*,?\s+&\s+{_NAME}|(?:,\s+{_NAME})*)?"
                      rf"(?:,\s+|(?:'s)?(?:\*\*)?\s+\()(\d{{4}})\b")


def _references() -> set[tuple[str, str]]:
    """(first author, year) of each entry in the methodology page's reference list."""
    refs = _read("docs/methodology.md").split("\n## References\n")[1]
    return set(re.findall(r"^\* ([^,( ]+)[^\n]*?\((\d{4})\)", refs, re.M))


def test_every_citation_in_the_docs_is_in_the_reference_list():
    entries = _references()
    for page in ["docs/methodology.md", *PAGES]:
        cited = set(CITATION.findall(_read(page).split("\n## References\n")[0]))
        assert cited <= entries, f"{page} cites {sorted(cited - entries)}, which the methodology references lack"


def test_design_decisions_cite_a_published_source():
    section = _section(_read("docs/methodology.md"), "## Visual design decisions")
    rows = [ln.strip("|").split("|") for ln in section.splitlines() if ln.startswith("| ")][1:]
    assert rows
    for decision, _reason, source in rows:
        assert CITATION.search(source), f"design decision {decision.strip()!r}: {source.strip()!r} is not a citation"


def test_each_statistical_method_cites_its_source():
    section = _section(_read("docs/methodology.md"), "## Statistical methods")
    methods = {part.splitlines()[0]: part for part in section.split("\n### ")[1:]}
    assert len(methods) >= 10
    for title, text in methods.items():
        assert CITATION.search(text), f"methodology: {title!r} cites no source"
    # Scott's KDE bandwidth rule is from his 1992 book; the 1979 paper is the histogram rule
    assert ("Scott", "1992") in CITATION.findall(methods["Kernel density estimates"])


def test_raincloud_docs_say_every_cloud_has_the_same_height():
    from matplotlib.collections import PolyCollection

    df = pd.DataFrame({"g": ["narrow"] * 50 + ["wide"] * 50,
                       "y": [0.01 * i for i in range(50)] + [0.3 * i for i in range(50)]})
    clouds = [c for c in vc.raincloud(df, x="g", y="y").axes.collections if isinstance(c, PolyCollection)]
    heights = {round(float(v[:, 1].max() - v[:, 1].min()), 9) for v in (c.get_paths()[0].vertices for c in clouds)}
    assert len(clouds) == 2 and len(heights) == 1  # a spread 30 times as wide gets a cloud of the same height
    assert "same peak height" in inspect.getdoc(vc.raincloud)
    kde = _section(_read("docs/methodology.md"), "### Kernel density estimates")
    assert "same peak height" in kde and "ridgeplot" in kde


def test_changelog_spearman_entry_names_the_sizes_that_got_a_false_star():
    text = _read("CHANGELOG.md")
    entry = text[text.index("- Spearman p-values for up to 9 pairs"):].split("\n- ")[0]
    named = set()
    for low, high, other in re.findall(r"n = (\d+)(?:–(\d+))?(?: or (\d+))?", entry):
        named |= set(range(int(low), int(high or low) + 1)) | ({int(other)} if other else set())
    # perfect rank agreement: the exact two-sided p is 2/n!, which stays at or above 0.05 only for n = 3 and 4
    exact = {n: stats.correlation_test(range(n), range(n), method="spearman")["p"] for n in range(3, 10)}
    assert named == {n for n, p in exact.items() if p >= 0.05}, entry


# --- migration page ------------------------------------------------------------------------------------------------

# The scripts that version 1.0 replaced, as they were at the last commit before the rebuild.
OLD_SCRIPTS_COMMIT = "a34caa60dc11ddfbcc097a3ccf450ab53912b71e"
OLD_SCRIPTS = (
    [f"analyses/{n}.py" for n in (
        "all_bars all_boxes all_scatters animated_bubble bar_time bench_bar centered_barplot compare_correlations "
        "correlogram divergent_bar donut_charts duration_plot grid_percent heatmap_calendar histo_group lollipop_plot "
        "nested_pie network_map pairs_plot plot_pca quadrant_norm significant_means stacked_percentages "
        "timeseries_fill").split()]
    + [f"templates/create_{n}.py" for n in (
        "barplot boxjitter boxplot bubbleplot bulletchart choropleth circularbar densityplot dumbbellplot funnelplot "
        "gantt heatmap histogram markermap radarplot ridgeplot sankeydiagram scatterplot stockplot venn violinplot "
        "waffle waterfallchart wordcloud").split()]
)


def test_migration_page_maps_every_old_script():
    text = _read("docs/migration.md")
    assert len(OLD_SCRIPTS) == 48
    missing = [path for path in OLD_SCRIPTS if path.split("/")[-1] not in text]
    assert not missing, f"docs/migration.md does not map {missing}"


def test_migration_page_and_readme_link_the_old_scripts():
    for page in ("docs/migration.md", "README.md"):
        assert f"https://github.com/elkronos/viz_calc/tree/{OLD_SCRIPTS_COMMIT}" in _read(page), page


def test_old_scripts_list_matches_the_commit():
    import subprocess

    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", OLD_SCRIPTS_COMMIT],
                             capture_output=True, text=True)
    except OSError:
        pytest.skip("git is not installed")
    if out.returncode != 0:
        pytest.skip("the commit is not in this checkout")
    scripts = [p for p in out.stdout.splitlines() if p.startswith(("analyses/", "templates/"))]
    assert sorted(scripts) == sorted(OLD_SCRIPTS)
