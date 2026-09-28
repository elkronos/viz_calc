# Migrating from the old scripts

Before version 1.0, viz_calc was a folder of standalone scripts
(`analyses/*.py`, `templates/*.py`). Each script ran a demo when imported,
most called `plt.show()` and returned nothing, several modified the caller's
DataFrame, and some blocked on `input()`, downloaded data or wrote files to
the working directory. Version 1.0 is an installable package with one
calling convention, tests, and documentation.

This page maps every old script to its replacement and lists the defects the
review found.

## Old → new

| Old script | Now | Notes |
|---|---|---|
| `analyses/significant_means.py` | `estimation_plot` | Welch instead of Student *t*; effect size + CI; raw data shown; any number of groups |
| `analyses/bench_bar.py` | `benchmark_bar` | CI error bars; groups classified by whether the CI excludes the benchmark |
| `analyses/lollipop_plot.py` | `lollipop` | |
| `analyses/divergent_bar.py` | `divergent_bar` | Absolute-value axis labels |
| `analyses/centered_barplot.py` | `centered_bar` | Wilson intervals |
| `analyses/correlogram.py` | `correlogram` | Multiple-comparison adjustment; CIs; pairwise *n* |
| `analyses/compare_correlations.py` | `compare_correlations` | Fisher z-tests instead of unsigned differences |
| `analyses/quadrant_norm.py` | `quadrant_plot` | Quadrant labels fixed; r with CI |
| `analyses/histo_group.py`, `templates/create_histogram.py` | `histogram` | Shared bins, per-group reference lines |
| `templates/create_ridgeplot.py` | `ridgeplot` | |
| `templates/create_boxjitter.py`, `create_boxplot.py`, `create_violinplot.py` | `raincloud` | One chart showing box, density and data |
| `templates/create_waffle.py` | `waffle` | Largest-remainder tiles |
| `analyses/grid_percent.py` (`grid_pecent`) | `percent_grid` | Wilson CI; any binary coding |
| `analyses/stacked_percentages.py` | `stacked_percentages` | |
| `analyses/donut_charts.py` | `donut_grid` | Consistent colours; no 8-chart limit |
| `analyses/nested_pie.py` | `nested_pie` | Long-format input; labels always match wedges |
| `templates/create_circularbar.py` | `circular_bar` | |
| `templates/create_waterfallchart.py` | `waterfall` | Matplotlib; changes or levels; proper totals |
| `templates/create_funnelplot.py` | `funnel` | Step and overall conversion rates |
| `templates/create_bulletchart.py` | `bullet` | Takes a DataFrame |
| `templates/create_venn.py` | `upset` | Any number of sets |
| `analyses/bar_time.py` | `period_bars` | Calendar periods; empty periods kept |
| `analyses/timeseries_fill.py` | `timeseries_fill` | Shading interpolated at crossings |
| `analyses/heatmap_calendar.py` | `calendar_heatmap` | No `calmap` dependency |
| `templates/create_gantt.py` | `gantt` | |
| `analyses/duration_plot.py` | `duration_plot` | |
| `analyses/animated_bubble.py` | `animated_bubble` | One size scale; fixed axes; no ipywidgets/adjustText |
| `analyses/plot_pca.py` | `pca_plot`, `pca` | Standardization; correct ellipses; biplot |
| `templates/create_radarplot.py` | `radar` | Aggregates and rescales metrics |
| `analyses/network_map.py` | `network_map` | Metrics computed from the graph; built-in Louvain |
| `templates/create_sankeydiagram.py` | `sankey` | |
| `analyses/all_bars.py` | `profile_bars` | Pages of small multiples, no `input()` |
| `analyses/all_boxes.py` | `profile_boxes` | Categorical × numeric pairs, ranked by η² |
| `analyses/all_scatters.py` | `profile_scatters` | Ranked by \|r\|, adjusted p-values |
| PowerPoint export in the `all_*` scripts | `to_pptx` | Aspect ratio preserved |

### Removed: thin wrappers with nothing to add

These scripts only renamed a single call to another library, and most had
defects. Use the library directly:

| Old script | Use instead |
|---|---|
| `templates/create_barplot.py` | `benchmark_bar`/`lollipop`, or `seaborn.barplot` / `plotly.express.bar` |
| `templates/create_scatterplot.py` | `quadrant_plot`, or `plotly.express.scatter(df, x, y, color=, symbol=, facet_col=)` |
| `templates/create_densityplot.py` | `ridgeplot`, `histogram(..., density_curve=True)`, or `seaborn.kdeplot` |
| `templates/create_heatmap.py` | `seaborn.histplot(df, x=, y=)` or `plotly.express.density_heatmap` |
| `analyses/pairs_plot.py` | `seaborn.pairplot(df, hue=...)`; for statistics, `correlogram` |
| `templates/create_choropleth.py` | `plotly.express.choropleth(df, geojson=, locations=, color=)` |
| `templates/create_markermap.py` | `folium.CircleMarker` |
| `templates/create_stockplot.py` | `plotly.graph_objects.Candlestick` or `period_bars` |
| `templates/create_wordcloud.py` | `wordcloud.WordCloud(...).generate(text)` |

## Defects found in the review

**Wrong results**

* `plot_pca`: took the square root of the eigenvalues twice when sizing
  ellipses, and used an unexplained 68% level, so ellipse sizes had no
  meaning. PCA ran on unscaled features, and axes lacked variance explained.
* `quadrant_norm`: quadrant percentages were printed in the wrong corners
  (the low-x/low-y share appeared top-left).
* `create_choropleth`: ignored the user's data and plotted random numbers.
* `nested_pie`: the example's inner labels (A1…A6, B1…B6) did not match the
  row-major wedge order, so wedges were mislabelled.
* `donut_charts`: colours were taken from a shared cycle, so after small
  wedges were dropped, colours no longer matched the legend.
* `create_waterfallchart`: treated the final "End" row as another increment
  instead of a total, and labelled bars with names instead of values.
* `create_histogram`: passed a bin **width** as a bin **count** to Plotly.
* `compare_correlations`: showed unsigned differences with no test of
  whether they exceed sampling noise.
* `correlogram`: starred raw p-values without multiple-comparison correction.
* `significant_means`: Student's *t*-test (assumes equal variances), fixed
  group labels "Group 1/2", and layout offsets in data units that broke for
  large or small values.
* `animated_bubble`: bubble sizes were normalized separately within every
  frame and colour group; weekly/monthly intervals only matched rows dated
  exactly on the period boundary, giving empty frames.
* `all_boxes`: selected columns by maximum value ≤ 100 and paired numeric
  columns with each other.
* `create_boxjitter`: ignored `palette`, and the legend labelled categories
  as "Box Plot" and "Jitter Points".
* `create_scatterplot`: `symbol_col` applied one symbol sequence across
  faceted traces.
* `network_map`: node size and colour columns were never used (NetworkX
  nodes have no such attributes), and `interactive` did nothing.

**Crashes**

* `lollipop_plot`: `KeyError: 'label_color'` with `labels=True` and no threshold.
* `create_waffle`: `IndexError` whenever rounding allocated more tiles than the grid.
* `create_heatmap`: failed without `facet_by` (grouped by `None`) and used
  the positional `DataFrame.pivot` signature removed in pandas 2.
* `create_histogram`: failed without `facet_by`.
* `create_venn`: failed with one set.
* `pairs_plot`: `np` was never imported, so `log_scale=True` crashed; and it
  only loaded seaborn sample datasets despite documenting URLs and paths.
* `network_map`: used `titleside`/`titlefont_size`, which current Plotly rejects (removed in Plotly 6).
* `create_densityplot`: used `shade=`, deprecated since seaborn 0.12 and scheduled for removal.
* `bar_time`, `animated_bubble`: used frequency aliases `"M"`, `"Q"`, `"Y"`,
  deprecated in pandas 2.2 and rejected by pandas 3.

**Side effects and usability**

* Every module ran its example when imported: opening windows, printing
  debug output, prompting for input, downloading GeoJSON, and writing
  `pairplot.png`, `rescaled_plot.png`, `wordcloud.png` and HTML files.
* Several functions modified the caller's DataFrame (`bar_time`,
  `grid_pecent`, `timeseries_fill`, `create_heatmap`, `create_gantt`,
  `create_stockplot`).
* Errors were printed and swallowed (`heatmap_calendar`, `pairs_plot`,
  `all_*`) or raised as `assert`.
* Global style state was changed (`sns.set`, `sns.set_theme`,
  `sns.set_style`), altering every later plot in the session.
* No package metadata, dependency list, tests or licence. The README had
  typos and names that did not match the code (`grid_pecent`,
  `create_dumbellplot`, `pairs_plots`) and omitted `create_bulletchart`.
