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

Every chart function takes a DataFrame first (`upset` also accepts a dict of
sets) and column names as keywords, and returns a
[`VizResult`](api.md#viz_calc.VizResult):

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

Functions that draw a single chart accept `ax=`, so several can share one
figure:

```python
import matplotlib.pyplot as plt
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
vc.lollipop(trial, x="arm", y="score", ax=left)
vc.centered_bar(trial, x="arm", y="score", threshold=55, ax=right)
fig.tight_layout()
```

`radar` and `circular_bar` draw on a **polar** Axes, so create that panel with
`projection="polar"` (or `plt.subplots(subplot_kw={"projection": "polar"})`);
an ordinary Axes raises `ValueError`:

```python
fig = plt.figure(figsize=(12, 5))
bars = fig.add_subplot(1, 2, 1)
spider = fig.add_subplot(1, 2, 2, projection="polar")
meas = datasets.measurements()
vc.lollipop(meas, x="group", y="mass", ax=bars)
vc.radar(meas, metrics=["length", "width", "depth", "mass"], group="group", ax=spider)
```

`bullet` draws one panel per row, so it takes `ax=` only for a single row.

These functions have no `ax=` and always create their own figure:

| Function | Why |
|---|---|
| `estimation_plot` | two linked panels: the data and the differences |
| `histogram` | lays out its own grid, one panel per facet, under a figure title; without `facet` the grid is 1 × 1 (`res.axes` is a 1 × 1 array) |
| `compare_correlations` | one heatmap per pair of groups |
| `percent_grid`, `donut_grid` | one panel per facet or row |
| `upset` | intersection bars, membership matrix and set sizes on linked axes |
| `calendar_heatmap` | one panel per year |
| `profile_bars`, `profile_boxes`, `profile_scatters` | pages of small multiples; `res.figure` is a list |
| `animated_bubble` | the animation redraws its own figure |
| `sankey` | returns a Plotly figure, as does `network_map(interactive=True)` |

## Using the statistics without a chart

All the methods are public in [`viz_calc.stats`](api.md#statistics):

```python
from viz_calc import stats

control_scores = trial.loc[trial["arm"] == "control", "score"]
treated_scores = trial.loc[trial["arm"] == "high dose", "score"]

stats.wilson_ci(12, 40)                              # (0.181, 0.454)
stats.welch_test(control_scores, treated_scores)     # treated − control: 4.6, 95% CI 0.2 to 9.1, p = 0.043
stats.adjust_pvalues([0.01, 0.04, 0.03], "holm")     # [0.03, 0.06, 0.06]
stats.compare_correlations_test(0.5, 100, 0.3, 100)  # z ≈ 1.67, p ≈ .095
```

## Colours

The default categorical palette is Okabe–Ito, designed to stay distinct for
colour-blind readers. For more than eight categories viz_calc switches to
`viridis`. `vc.palette(n, colors)` gives you the same colours for your own
additions. How you change a chart's colours depends on what it colours:

| Argument | Takes | Functions |
|---|---|---|
| `colors=` | a Matplotlib colormap name (e.g. `"viridis"`) or a list of colours, cycled if short; one colour per group or category | `estimation_plot`, `histogram`, `ridgeplot`, `raincloud`, `waffle`, `stacked_percentages`, `donut_grid`, `nested_pie`, `circular_bar`, `gantt`, `animated_bubble`, `pca_plot`, `radar` |
| `colors=` | a diverging colormap name (default `"RdBu"`) or a list of exactly one colour per level, most negative first | `likert` |
| `colors=` | a tuple with one colour per role, in this order (not a colormap name) | `benchmark_bar` (below, above, indistinguishable), `waterfall` (increase, decrease, total), `dumbbell` (start, end), `divergent_bar` (left, right), `centered_bar` (at or above, below), `percent_grid` (success, other), `timeseries_fill` (first series, second series), `duration_plot` (outer, inner) |
| `color=` | one colour | `lollipop`, `quadrant_plot`, `funnel`, `upset`, `period_bars`, `profile_bars` |
| `bar_color=` | one colour for the measure; the bands are shades of grey | `bullet` |
| `cmap=` | a colormap name | `correlogram`, `compare_correlations`, `calendar_heatmap` (plus `missing_color=` for days without data) |
| none | the default palette | `network_map` (one colour per community), `sankey`, `profile_boxes`, `profile_scatters` |

In `animated_bubble`, `color=` names the **column** whose categories set the
bubble colours; the colours themselves go in `colors=`.

## Conventions

Arguments are named for the role a column plays, and most names mean the
same thing everywhere. The exceptions are listed here.

* **`x` and `y`.** In the group comparisons (`estimation_plot`,
  `benchmark_bar`, `lollipop`, `centered_bar`, `raincloud`), `x` is the
  grouping column and `y` the numeric column being compared, whichever way
  the chart is drawn: `raincloud` and `lollipop` (with its default
  `horizontal=True`) list the groups down the vertical axis. In `histogram`
  and `ridgeplot`, `x` is the numeric column whose distribution is drawn, and
  the groups come from `hue=`/`facet=` or `group=`. In `quadrant_plot` and
  `animated_bubble`, `x` and `y` are the horizontal and vertical axes.
* **`group`** is a categorical column whose levels are drawn separately, as
  colours, rows, sections or panels (`ridgeplot`, `compare_correlations`,
  `stacked_percentages`, `circular_bar`, `gantt`, `pca_plot`, `radar`).
  `histogram` splits by `hue=` (overlaid) and `facet=` (panels), and
  `percent_grid` by `facet=`.
* **`value`** is the numeric column being drawn (`waffle`, `nested_pie`,
  `circular_bar`, `waterfall`, `funnel`, `bullet`, `period_bars`,
  `calendar_heatmap`, `sankey`). Where it is optional (`waffle`,
  `nested_pie`, `calendar_heatmap`), leaving it out counts rows.
* **`label`** is the column that names each bar or row (`dumbbell`,
  `circular_bar`, `waterfall`, `bullet`, `duration_plot`); in
  `quadrant_plot` and `animated_bubble` it is an optional column that
  annotates the points.
* **`columns`** is a list of column names to use (`correlogram`,
  `compare_correlations`, `donut_grid`, `profile_bars`, `profile_scatters`;
  `None` means every suitable column), except in `waffle`, where `rows` and
  `columns` are the size of the grid in tiles.
* **`labels`** switches the text labels on the chart on or off (`lollipop`,
  `centered_bar`, `likert`, `donut_grid`, `nested_pie`, `network_map`,
  `profile_bars`), except in `timeseries_fill`, where it is a list of the two
  legend names. Other two-part legends are named in pairs: `start_label` /
  `end_label` (`dumbbell`), `left_label` / `right_label` (`divergent_bar`)
  and `outer_name` / `inner_name` (`duration_plot`).
* **`alpha`** is the significance level in `correlogram` and
  `compare_correlations`, but the opacity of the bars or shading in
  `histogram` and `timeseries_fill`.
* **`level`** is the confidence level of intervals (default 0.95); `levels`
  in `likert` is the response scale.
* **Order.** Where a function has `order=`, it fixes the order of the categories
  (otherwise categories keep their Categorical order or first appearance). Where it applies to
  one particular argument it is named after it: `hue_order`/`facet_order` in
  `histogram`, `facet_order` in `percent_grid`, and
  `group_order`/`category_order` in `stacked_percentages`.
* **Grid width** of multi-panel figures is `col_wrap` in `histogram`,
  `percent_grid` and `donut_grid`, and `ncols` in the `profile_*` functions.

## Example datasets

`viz_calc.datasets` generates small, seeded, synthetic datasets (no
downloads, same numbers every time): `trial`, `survey`, `sales`,
`measurements`, `projects`, `network`, `memberships`. They are used
throughout these docs.
