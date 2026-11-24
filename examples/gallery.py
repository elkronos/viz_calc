"""Render every chart in the documentation gallery to docs/images/.

Run from the repository root:  python examples/gallery.py [name ...]
Each entry is also a minimal, copy-pasteable example of one function.
Given names, only those charts are rendered. Without names, every chart is
rendered, docs/gallery.md is rewritten and images of removed examples are
deleted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import viz_calc as vc  # noqa: E402
from viz_calc import datasets  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"

trial = datasets.trial()
survey = datasets.survey()
sales = datasets.sales()
meas = datasets.measurements()
proj = datasets.projects()
edges = datasets.network()
survey["work_mode"] = survey["remote"].map({True: "Remote", False: "On-site"})
likert_items = ["Workload is manageable", "My manager supports me", "Tools are adequate",
                "I see a future here", "Meetings are useful"]
levels = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
monthly = (sales.assign(month=sales["date"].dt.to_period("M").astype(str))
           .groupby("month", as_index=False)[["online", "store"]].sum())
# made-up Gapminder-style data: six countries every five years (GDP per person in $1,000s, population in millions)
start = pd.DataFrame({"country": ["A", "B", "C", "D", "E", "F"],
                      "continent": ["Africa", "Africa", "Asia", "Asia", "Europe", "Europe"],
                      "gdp": [1.5, 3.0, 4.0, 9.0, 25.0, 35.0], "life_exp": [55.0, 60.0, 66.0, 71.0, 77.0, 80.0],
                      "population": [30.0, 60.0, 90.0, 140.0, 10.0, 70.0]})
countries = pd.concat([start.assign(year=year, gdp=start["gdp"] * 1.04 ** (year - 2000),
                                    life_exp=start["life_exp"] + 0.25 * (year - 2000),
                                    population=start["population"] * 1.01 ** (year - 2000))
                       for year in range(2000, 2021, 5)], ignore_index=True)

EXAMPLES = {
    # compare
    "estimation_plot": lambda: vc.estimation_plot(trial, x="arm", y="score"),
    "benchmark_bar": lambda: vc.benchmark_bar(trial, x="arm", y="score", threshold=50),
    "lollipop": lambda: vc.lollipop(survey, x="team", y="tenure_years", stat="median", threshold=3),
    "dumbbell": lambda: vc.dumbbell(monthly.tail(8), label="month", start="store", end="online",
                                      sort=False, show_values=True),
    "divergent_bar": lambda: vc.divergent_bar(monthly.tail(8), category="month", left="store", right="online"),
    "centered_bar": lambda: vc.centered_bar(trial, x="arm", y="score", threshold=55),
    "likert": lambda: vc.likert(survey, items=likert_items, levels=levels),
    # correlation
    "correlogram": lambda: vc.correlogram(meas, method="spearman"),
    "compare_correlations": lambda: vc.compare_correlations(meas, group="group", columns=["length", "width", "depth", "mass"]),
    "quadrant_plot": lambda: vc.quadrant_plot(trial, x="baseline", y="score"),
    # distribution
    "histogram": lambda: vc.histogram(trial, x="score", hue="arm", facet="site", stat="density", ref_line="mean"),
    "ridgeplot": lambda: vc.ridgeplot(meas, x="depth", group="group"),
    "raincloud": lambda: vc.raincloud(trial, x="arm", y="score"),
    # composition
    "waffle": lambda: vc.waffle(survey, category="team"),
    "percent_grid": lambda: vc.percent_grid(trial, column="improved", facet="arm"),
    "stacked_percentages": lambda: vc.stacked_percentages(survey, group="team", category="Meetings are useful",
                                                          category_order=levels, colors="RdBu", horizontal=True),
    "donut_grid": lambda: vc.donut_grid(pd.crosstab(survey["team"], survey["work_mode"])),
    "nested_pie": lambda: vc.nested_pie(survey, outer="team", inner="work_mode"),
    "circular_bar": lambda: vc.circular_bar(monthly.assign(year=monthly["month"].str[:4]), label="month",
                                            value="online", group="year", sort=False),
    "waterfall": lambda: vc.waterfall(pd.DataFrame({"item": ["Opening", "Sales", "Refunds", "Costs", "Tax"],
                                                    "amount": [1000, 650, -120, -380, -45]}),
                                      label="item", value="amount", start_label="Opening", total_label="Closing"),
    "funnel": lambda: vc.funnel(pd.DataFrame({"stage": ["Visited", "Signed up", "Activated", "Paid"],
                                              "users": [10000, 3200, 1900, 410]}), stage="stage", value="users"),
    "bullet": lambda: vc.bullet(pd.DataFrame({"kpi": ["Revenue", "Profit", "Satisfaction"],
                                              "value": [270, 22, 4.1], "target": [250, 26, 4.5],
                                              "poor": [150, 10, 2.5], "ok": [225, 20, 3.5], "good": [300, 30, 5]}),
                                label="kpi", value="value", target="target", bands=["poor", "ok", "good"]),
    "upset": lambda: vc.upset(datasets.memberships(), max_intersections=15),
    # time
    "period_bars": lambda: vc.period_bars(sales, date="date", value="online", freq="month", stat="sum"),
    "timeseries_fill": lambda: vc.timeseries_fill(sales.set_index("date")[["online", "store"]].resample("W").mean()
                                                  .reset_index(), time="date", series=["online", "store"]),
    "calendar_heatmap": lambda: vc.calendar_heatmap(sales, date="date", value="online"),
    "gantt": lambda: vc.gantt(proj, task="task", start="start", end="end", group="team"),
    "duration_plot": lambda: vc.duration_plot(proj, label="task", start="start", end="end",
                                              inner_start="work_start", inner_end="work_end"),
    "animated_bubble": lambda: vc.animated_bubble(countries, time="year", x="gdp", y="life_exp", size="population",
                                                  color="continent", label="country"),
    # multivariate
    "pca_plot": lambda: vc.pca_plot(meas, features=["length", "width", "depth", "mass"], group="group", loadings=True),
    "radar": lambda: vc.radar(meas, metrics=["length", "width", "depth", "mass"], group="group"),
    # network
    "network_map": lambda: vc.network_map(edges, source="source", target="target", weight="strength", size_by="betweenness"),
    # explore
    "profile_bars": lambda: vc.profile_bars(meas, max_levels=10),
    "profile_boxes": lambda: vc.profile_boxes(meas, max_levels=10),
    "profile_scatters": lambda: vc.profile_scatters(meas, method="pearson", p_adjust="holm"),
}


SECTIONS = {
    "Comparing groups": ["estimation_plot", "benchmark_bar", "lollipop", "dumbbell", "divergent_bar", "centered_bar",
                         "likert"],
    "Correlation": ["correlogram", "compare_correlations", "quadrant_plot"],
    "Distributions": ["histogram", "ridgeplot", "raincloud"],
    "Part-to-whole, flows and sets": ["waffle", "percent_grid", "stacked_percentages", "donut_grid", "nested_pie",
                                      "circular_bar", "waterfall", "funnel", "bullet", "upset"],
    "Time and projects": ["period_bars", "timeseries_fill", "calendar_heatmap", "gantt", "duration_plot",
                          "animated_bubble"],
    "Multivariate and networks": ["pca_plot", "radar", "network_map"],
    "Exploring a dataset": ["profile_bars", "profile_boxes", "profile_scatters"],
}

# A line under the image, for charts whose image needs one.
NOTES = {
    "animated_bubble": 'The image is the first of five frames; `res.info["animation"]` plays them all.',
}


def write_markdown() -> None:
    """Write docs/gallery.md from the EXAMPLES source so the page never drifts from the code."""
    import ast
    import textwrap

    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    snippets = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "EXAMPLES":
            for key, value in zip(node.value.keys, node.value.values):
                body = ast.get_source_segment(src, value.body)
                first, *rest = body.splitlines()
                rest = textwrap.indent(textwrap.dedent("\n".join(rest)), "    ") if rest else ""
                snippets[key.value] = first + ("\n" + rest if rest else "")
    setup = src[src.index("trial = datasets.trial()"):src.index("EXAMPLES = {")].strip()
    lines = ["# Gallery", "",
             "Every chart below is produced by `examples/gallery.py`; this page is generated from it.",
             "All examples use this setup:", "", "```python", "import viz_calc as vc",
             "from viz_calc import datasets", "import pandas as pd", "", setup, "```", ""]
    for section, names in SECTIONS.items():
        lines += [f"## {section}", ""]
        for name in names:
            lines += [f"### `{name}`", "", "```python", f"res = {snippets[name]}", "```", "",
                      f"![{name}](images/{name}.png)", ""]
            if name in NOTES:
                lines += [NOTES[name], ""]
    (OUT.parent / "gallery.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote gallery.md")


def main(names: list[str]) -> None:
    unknown = sorted(set(names) - set(EXAMPLES))
    if unknown:
        raise SystemExit(f"unknown example(s) {unknown}; choose from {sorted(EXAMPLES)}")
    OUT.mkdir(parents=True, exist_ok=True)
    if not names:
        write_markdown()
        for stale in sorted(OUT.glob("*.png")):
            if stale.stem not in EXAMPLES:
                stale.unlink()
                print(f"removed {stale.name}")
    for name, make in EXAMPLES.items():
        if names and name not in names:
            continue
        res = make()
        pages = res.figure if isinstance(res.figure, list) else [res.figure]
        if len(pages) > 1:  # one image per chart: an example must fit on one page
            raise SystemExit(f"{name}: the example draws {len(pages)} pages; the gallery shows one")
        pages[0].savefig(OUT / f"{name}.png", dpi=110, bbox_inches="tight")
        plt.close("all")
        print(f"wrote {name}.png")


if __name__ == "__main__":
    main(sys.argv[1:])
