# viz_calc

**Visual calculators for pandas data.** Each function draws a chart *and*
returns the numbers behind it, computed with published, citable methods:
Welch tests and bootstrap effect sizes, Wilson intervals for percentages,
multiplicity-adjusted correlation matrices, Fisher z-tests for group
differences in correlation, and more.

📖 **Documentation, walkthroughs and gallery:** https://elkronos.github.io/viz_calc/

```python
import viz_calc as vc
from viz_calc import datasets

trial = datasets.trial()
res = vc.estimation_plot(trial, x="arm", y="score")

res.figure                  # Matplotlib figure
res.table                   # n, mean, SD and 95% CI per group
res.info["comparisons"]     # mean difference + bootstrap CI, Hedges' g, Holm-adjusted Welch p
res.save("effect.png")
```

<p align="center"><img src="docs/images/estimation_plot.png" width="560" alt="Estimation plot"></p>

## Install

```bash
pip install "viz_calc @ git+https://github.com/elkronos/viz_calc"
pip install "viz_calc[all] @ git+https://github.com/elkronos/viz_calc"   # + networkx, plotly, python-pptx
```

The core needs only NumPy, pandas, Matplotlib and SciPy (Python ≥ 3.10).

## What it offers beyond Matplotlib, seaborn and Plotly

| Common practice | viz_calc |
|---|---|
| Bar of means ± SE and a star | Raw data + mean CI + effect size with bootstrap CI (`estimation_plot`) |
| Student's t-test | Welch's t-test by default |
| Starring every p < .05 in a correlation matrix | Holm / Benjamini–Hochberg adjustment across unique pairs (`correlogram`) |
| Eyeballing two groups' heatmaps | Fisher r-to-z test for each difference (`compare_correlations`) |
| "65%" with no interval | Wilson score interval (`percent_grid`, `centered_bar`) |
| Venn diagrams | UpSet plots for any number of sets (`upset`) |
| Rounded waffle tiles that overflow | Largest-remainder apportionment (`waffle`) |
| Plots that pop up, print and write files | No side effects; results returned as data |

Every choice is referenced on the [Methodology](https://elkronos.github.io/viz_calc/methodology/) page.

## Functions

| Area | Functions |
|---|---|
| Comparing groups | `estimation_plot`, `benchmark_bar`, `lollipop`, `dumbbell`, `divergent_bar`, `centered_bar`, `likert` |
| Correlation | `correlogram`, `compare_correlations`, `quadrant_plot` |
| Distributions | `histogram`, `ridgeplot`, `raincloud` |
| Part-to-whole, flows, sets | `waffle`, `percent_grid`, `stacked_percentages`, `donut_grid`, `nested_pie`, `circular_bar`, `waterfall`, `funnel`, `bullet`, `upset` |
| Time and projects | `period_bars`, `timeseries_fill`, `calendar_heatmap`, `gantt`, `duration_plot`, `animated_bubble` |
| Multivariate and networks | `pca`, `pca_plot`, `radar`, `network_map`, `sankey` |
| Exploring a dataset | `profile_bars`, `profile_boxes`, `profile_scatters`, `to_pptx` |
| Statistics only | `viz_calc.stats`: `wilson_ci`, `mean_ci`, `welch_test`, `hedges_g`, `bootstrap_ci`, `adjust_pvalues`, `correlation_test`, `compare_correlations_test`, `histogram_bins`, `largest_remainder` |

See the [gallery](https://elkronos.github.io/viz_calc/gallery/) for every chart with its code.

## Upgrading from the old scripts

The `analyses/` and `templates/` scripts have been replaced by the package.
[Migrating from the old scripts](https://elkronos.github.io/viz_calc/migration/)
maps each script to its replacement and lists the defects fixed.

## Development

```bash
pip install -e ".[dev]"
pytest                    # unit tests, including checks against SciPy and worked examples
ruff check src tests
python examples/gallery.py && mkdocs serve    # rebuild images and preview the docs (needs ".[docs]")
```

## License

MIT © elkronos
