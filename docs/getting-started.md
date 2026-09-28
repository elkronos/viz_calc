# Getting started

## Install

```bash
pip install "viz_calc @ git+https://github.com/elkronos/viz_calc"
```

Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `network` | networkx | `network_map` |
| `interactive` | plotly | `sankey`, `network_map(interactive=True)` |
| `pptx` | python-pptx | `to_pptx` |
| `all` | all of the above | |

```bash
pip install "viz_calc[all] @ git+https://github.com/elkronos/viz_calc"
```

Python 3.10+ with NumPy ≥ 1.24, pandas ≥ 2.0, Matplotlib ≥ 3.7 and SciPy ≥ 1.11.

## The one pattern to learn

Every function takes a DataFrame first and column names as keywords, and
returns a [`VizResult`](api.md#viz_calc.VizResult):

```python
import viz_calc as vc
from viz_calc import datasets

trial = datasets.trial()                      # seeded example data: 3 arms × 40 people
res = vc.benchmark_bar(trial, x="arm", y="score", threshold=50)

res.figure      # matplotlib.figure.Figure
res.axes        # the Axes you can keep customising
res.table       # the numbers that were drawn
res.info        # extra results and the settings used
res.save("benchmark.png")   # PNG, SVG or PDF; HTML for Plotly figures
```

`res.table` for the call above:

| arm | n | mean | sd | se | ci_low | ci_high | status |
|---|---|---|---|---|---|---|---|
| control | 40 | 48.91 | 9.10 | 1.44 | 46.00 | 51.82 | indistinguishable |
| low dose | 40 | 52.26 | 16.14 | 2.55 | 47.09 | 57.42 | indistinguishable |
| high dose | 40 | 53.53 | 10.86 | 1.72 | 50.05 | 57.00 | above |

The table is an ordinary DataFrame: export it with `res.table.to_csv(...)`,
cite it in a report, or test it.

## Showing figures

viz_calc never calls `plt.show()`. In Jupyter the figure appears
automatically when it is the last thing in a cell (`res.figure`). In a script:

```python
import matplotlib.pyplot as plt
res = vc.waffle(datasets.survey(), category="team")
plt.show()
```

## Putting charts into your own layouts

Single-panel functions accept `ax=`:

```python
import matplotlib.pyplot as plt
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
vc.lollipop(trial, x="arm", y="score", ax=left)
vc.centered_bar(trial, x="arm", y="score", threshold=55, ax=right)
fig.tight_layout()
```

Multi-panel functions (`estimation_plot`, `histogram` with facets,
`compare_correlations`, `upset`, `percent_grid`, `bullet`, `calendar_heatmap`)
create their own figure.

## Using the statistics without a chart

All the methods are public in [`viz_calc.stats`](api.md#statistics):

```python
from viz_calc import stats

stats.wilson_ci(12, 40)                         # (0.181, 0.454)
stats.welch_test(control_scores, treated_scores)
stats.adjust_pvalues([0.01, 0.04, 0.03], "holm")
stats.compare_correlations_test(0.5, 100, 0.3, 100)   # z ≈ 1.67, p ≈ .095
```

## Colours

The default categorical palette is Okabe–Ito, designed to stay distinct for
colour-blind readers. For more than eight categories viz_calc switches to
`viridis`. Every function accepts `colors=` (a colormap name or a list);
`vc.palette(n, colors)` gives you the same colours for your own additions.

## Example datasets

`viz_calc.datasets` generates small, seeded, synthetic datasets (no
downloads, same numbers every time): `trial`, `survey`, `sales`,
`measurements`, `projects`, `network`, `memberships`. They are used
throughout these docs.
