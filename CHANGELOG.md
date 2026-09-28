# Changelog

## 1.1.0

Fixes from an adversarial review of 1.0.0, each covered by a regression test
in `tests/` and run on the newest and the oldest supported dependency
versions.

### Changed
- `stacked_percentages(category_order_=...)` is renamed `category_order=...`;
  the old name still works with a `FutureWarning`.
- Category orders (`waffle`, `stacked_percentages`) must list every level that
  occurs, may add unused levels (drawn as zero) and may not repeat a level.
- Unknown correlation methods, a `success` value that is not one of a
  two-valued column's values, empty data, all-missing grouping columns,
  one-sided quadrant splits, missing or infinite edge weights, ambiguous date
  strings (e.g. day-first mixed with month-first), numbers passed as dates
  (years? epoch seconds?), columns mixing values with and without a UTC
  offset in the charts that keep time zones (`gantt`, `duration_plot`,
  `timeseries_fill`; calendar views place every value on its written local
  day), and non-positive
  `upset(max_intersections=...)` now raise clear errors.
- Number labels keep significant digits below 1 (`0.034`, not `0`); axes that
  reach 10,000 use K/M/B/T with the fewest decimals that show every tick
  exactly (`2.5K, 5.0K, 7.5K`), per-tick labels on log axes, and Matplotlib's
  offset notation for narrow ranges at large magnitudes.
- Optional column arguments are compared with `None`, so columns labelled `0`
  (e.g. from a headerless CSV) are no longer ignored.
- `level=` must be strictly between 0 and 1 (`0.95`); `level=95` raises
  instead of giving NaN intervals and misclassified `benchmark_bar` groups.
- Options and counts are checked before anything is drawn, and a bad value
  raises a `ValueError` listing the valid ones: `lollipop(sort=...)` (`"asc"`
  used to sort descending; use `None` for no sorting), `network_map(size_by=)`,
  `histogram(ref_line=, bins=, col_wrap=)`, `calendar_heatmap(stat=)` (mean,
  max and min need a value column), `p_adjust`, `per_page`, `rows`,
  `col_wrap`, `trend_window`, `upset(min_size=, max_intersections=)` and more.
- Fixed-role `colors` (`benchmark_bar`, `centered_bar`, `dumbbell`,
  `divergent_bar`, `waterfall`, `percent_grid`, `timeseries_fill`,
  `duration_plot`) must be a tuple of the documented roles; a colormap name
  raises instead of being read letter by letter (`benchmark_bar(colors=
  "viridis")` drew every bar red). `likert` needs one colour per level.
- Column-list arguments accept any list-like (a pandas Index, array, tuple)
  and reject a single string instead of splitting it into one-letter columns.
- An all-missing column raises "column ... has no non-missing values" in
  every function that needs values from it, instead of an internal error or
  an empty chart; one column passed for two roles (for
  example `dumbbell(start="a", end="a")`) raises a clear error.
- A plotting call that raises closes the figures it opened, so no half-drawn
  chart is left for `plt.show()` or Jupyter.
- Spearman p-values for up to 9 pairs are exact permutation p-values (the t
  approximation gave p = 0, and a star, for any perfect rank agreement at
  n = 3–5).
- Freedman–Diaconis bins on whole-number data use whole-number widths with
  edges on half-integers instead of a comb of empty bins.
- `radar` and `circular_bar` raise a clear error for a non-polar `ax`;
  `bullet` draws into `ax=` for a single row.
- Number labels choose K/M/B/T after rounding (999,999 reads `1M`, not
  `1000K`).

### Fixed
- Histograms fall back from Freedman–Diaconis to Sturges when the IQR is zero
  (or zero up to floating-point noise), or FD would need over 100,000 bins
  (previously one bin, or a MemoryError). Outliers keep FD unless they are
  extreme enough to hit that safety limit.
- `network_map`: repeated edges are summed instead of the last row winning;
  categorical node columns, non-string column labels and node order work;
  directed communities use summed reciprocal weights; repeated rows are
  summed exactly (or correctly rounded); weighted betweenness no longer
  lets rounding hand one of two equally short paths all the credit: weights
  that cannot have been rounded when stored (integers, 0.125) are compared
  exactly at any graph size, with floating point ordering the paths and
  near-ties settled exactly, and other floats within igraph's relative
  tolerance of 1e-10 (four machine epsilons for float32/float16), never
  merging a real extra hop; results do not depend on row order
  (`info["betweenness_arithmetic"]`, `info["betweenness_tolerance"]`);
  weights too small to invert, or repeated rows summing past the largest
  float, raise a clear error; directed graphs draw arrowheads (Matplotlib
  and Plotly).
- Dates: calendar views (`calendar_heatmap`, `period_bars`) place
  timezone-aware values by their local day, including mixed UTC offsets
  across DST; `gantt`, `duration_plot` and `timeseries_fill` keep time zones,
  so durations are true elapsed times and the repeated DST hour is not merged;
  mixed ISO date/date-time strings parse; string dates sort chronologically in
  `timeseries_fill`; numeric years (object or Categorical) stay numeric.
- `calendar_heatmap` shows all-missing days as no data, not zero.
- `upset` no longer counts missing values or `"0"` strings as membership.
- `radar` leaves a missing group metric as a gap (with vertex markers) and
  handles nullable dtypes; `pca_plot` keeps categorical group order with any
  index; `correlogram(hide_nonsignificant=True)` keeps `info["matrix"]`.
- `raincloud` draws densities for narrow groups; `bullet` keeps zero, negative
  measures and targets in view and accepts array bands; `waterfall` uses
  `start_label`; `gantt`/`animated_bubble` draw missing groups in grey with a
  legend entry; `animated_bubble` handles NaN and nullable columns and labels
  only drawn bubbles; `dumbbell` computes changes safely for booleans,
  unsigned/small and nullable integers while keeping int64 exact;
  `quadrant_plot` classifies points on the mean consistently whether or not it
  standardizes; `donut_grid` skips boolean columns.
- `to_pptx` keeps each image's true aspect ratio and puts one title per item
  on every page of multi-page results; `VizResult.save` returns the real path
  when no extension is given, accepts file-like objects and writes a
  multi-page result to one as a PDF.
- Nullable `Int64`/`Float64` columns holding `pd.NA` work on pandas 2.0 in the
  statistics functions, `correlogram`, `compare_correlations`,
  `profile_scatters`, `period_bars` (empty periods), `funnel`, `donut_grid`
  and `circular_bar`.
- `welch_test` gives t = −inf for a negative difference between constant
  groups, and groups such as `[0.1] * 3` count as constant; `mean_ci`,
  `welch_test` and `hedges_g` no longer overflow beyond about 1e154, and
  `largest_remainder`/`waffle` fill the grid (with correct percentages) when
  the values' sum overflows.
- `threshold="mean"` (`centered_bar`, `benchmark_bar`) and
  `quadrant_plot(center="mean")` use the correctly rounded mean, so a value
  on the mean is no longer counted below it.
- `nested_pie` works with a Categorical outer column; `stacked_percentages`
  works with repeated index labels on pandas 3; `upset` draws a single set and
  rejects the set names `size` and `degree`; `circular_bar` keeps rows with a
  missing group as "(missing)"; `funnel` and `waterfall` reject missing values
  (`waterfall` returned a NaN total) and `funnel` negative ones; `waffle`
  rejects negative rows and leaves out unused Categorical levels whether it
  counts or sums; `percent_grid` titles an empty group "no data (n=0)".
- `animated_bubble` plays text dates and ordered Categoricals in time or
  category order, with readable frame stamps and no "Animation was deleted"
  warning; `duration_plot` rejects windows that end before they start;
  `gantt` no longer warns on pandas 3 for Categorical groups with unused
  categories.
- Datetime category orders work on NumPy 1.x, and date groups read
  `2024-01-01` in labels, legends and titles.
- Two identical constant groups (`[0.1] * 3` and `[0.1] * 10`) are no longer
  called different: `welch_test` gives t = nan and p = 1, the bootstrap
  difference and its interval are exactly 0, and result tables report SD 0.
  `adjust_pvalues` rejects p-values outside [0, 1], and the statistics
  functions treat `pd.NA` in plain lists as missing.
- Histogram FD bins keep the 100,000-bin limit after rounding to whole-number
  widths, and a real IQR at large magnitudes (integers near 1e15) is no longer
  taken for zero.
- `nested_pie`, `waffle` and `pca_plot` show rows with a missing category or
  group as a grey "(missing)" entry instead of silently dropping them.
- `profile_boxes` orders numeric and Categorical levels naturally (1, 2, 10,
  not 1, 10, 2); `divergent_bar` follows Categorical order.
- `funnel` shows an undefined conversion rate as "–" instead of "nan%";
  `period_bars` no longer wraps around when summing very large integers;
  `bullet` reads integer band labels as column names when they are columns;
  `sankey` rejects missing flow values; `network_map` accepts a NumPy seed.
- `sankey` rejects rows with a missing source or target instead of silently
  leaving their flow out; `upset` no longer counts missing values as members
  when sets are given as a mapping.
- Columns labelled `False`/`True` (as a pivot on a boolean column gives) are
  selected as columns, not read as a row mask; `correlogram` and the other
  column-list functions reject a repeated column; `likert` rejects response
  levels named `item`, `n` or `net`.
- `percent_grid` rounds dots half up (2.5% shows 3 dots) and accepts a
  datetime `success` value on NumPy 1.x; `funnel` shows small conversions
  such as 0.45% instead of "0%"; a missing `bullet` band limit no longer
  hides the later bands; `waffle` integer sums no longer wrap around.
- `network_map` rejects one column in two roles, raises when path lengths
  overflow, and settles float ties from very heavy edges exactly without
  slowing down.
- Confidence levels in titles keep their digits (`97.5% CI`, not `98% CI`);
  a group or category column named like a table column (`n`, `status`,
  `percent`, `value`, ...) raises instead of being overwritten; SDs, SEs and
  densities are computed without overflow at extreme magnitudes;
  `profile_boxes` skips empty columns; `pca_plot` works with a MultiIndex;
  `histogram(bins=...)` is validated; `largest_remainder` sums exactly to
  large totals; `upset` accepts sets named `False`/`True`; `circular_bar` and
  `gantt` draw an all-missing group column as "(missing)".
- `network_map`: with inexact float weights, paths through an edge too short
  for float64 to add to a path length are no longer dropped; weights above
  2**256 no longer overflow Louvain or the layout, and node strengths past
  the largest float raise a clear error.
- Number labels beyond 999T use scientific notation; time-zone-aware
  midnight dates read `2024-01-01`; `quadrant_plot(standardize=True)` works
  at very large and very small magnitudes; `to_pptx` accepts a single title
  string.

### Documentation
- Getting started lists which functions take `colors=` (and in what form),
  `color=` or `cmap=`, which have no `ax=` and why, and that `radar` and
  `circular_bar` need a polar Axes; a new Conventions section explains the
  shared argument names.
- Walkthrough numbers match the code and every code block runs as written;
  a test runs each page and compares its tables.

## 1.0.0

First packaged release. The standalone `analyses/` and `templates/` scripts
are replaced by the installable `viz_calc` package.

### Added
- `VizResult`: every function returns the figure plus a tidy table of the
  values drawn and extra statistics.
- `viz_calc.stats`: Wilson intervals, t-based mean CIs, Welch's test, Hedges' g,
  BCa bootstrap intervals, Holm/BH/Bonferroni adjustment, correlation CIs,
  Fisher r-to-z comparison, histogram bin rules, largest-remainder apportionment.
- New charts: `estimation_plot`, `raincloud`, `likert`, `upset`, `percent_grid`
  (with CIs), `profile_*` small multiples, `to_pptx`.
- `viz_calc.datasets`: seeded synthetic example data.
- Tests (including minimum-dependency checks), GitHub Actions CI, and a
  documentation site with walkthroughs, gallery, methodology and API reference.

### Changed
- One calling convention: `func(data, column=..., ...)`; single-panel functions accept `ax=`.
- No side effects: nothing is shown, printed or written, and input data is never modified.
- Colour-blind-safe Okabe–Ito palette by default.

### Fixed
See docs/migration.md for the full list of defects in the old scripts,
including wrong PCA ellipses, misplaced quadrant labels, a choropleth that
ignored its input, waffle overflow crashes, and incompatibilities with
pandas 3 and Plotly 6.

### Removed
Thin wrappers that added nothing over the underlying library (bar, scatter,
density, 2-D heatmap, pairs plot, choropleth, marker map, stock plot, word
cloud). docs/migration.md gives the one-line replacement for each.
