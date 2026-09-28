# Changelog

## 1.1.0

Fixes from an adversarial review of 1.0.0, each covered by a regression test
(`tests/test_edge_cases.py`) and run on the newest and the oldest supported
dependency versions.

### Changed
- `stacked_percentages(category_order_=...)` is renamed `category_order=...`;
  the old name still works with a `FutureWarning`.
- Category orders (`waffle`, `stacked_percentages`) must list every level that
  occurs, may add unused levels (drawn as zero) and may not repeat a level.
- Unknown correlation methods, a `success` value that is not one of a
  two-valued column's values, empty data, all-missing grouping columns,
  one-sided quadrant splits, missing or infinite edge weights, ambiguous date
  strings (e.g. day-first mixed with month-first) and non-positive
  `upset(max_intersections=...)` now raise clear errors.
- Number labels keep significant digits below 1 (`0.034`, not `0`); axes that
  reach 10,000 use K/M/B/T with the fewest decimals that show every tick
  exactly (`2.5K, 5.0K, 7.5K`), per-tick labels on log axes, and Matplotlib's
  offset notation for narrow ranges at large magnitudes.
- Optional column arguments are compared with `None`, so columns labelled `0`
  (e.g. from a headerless CSV) are no longer ignored.

### Fixed
- Histograms fall back from Freedman–Diaconis to Sturges when the IQR is zero
  (or zero up to floating-point noise), or FD would need over 100,000 bins
  (previously one bin, or a MemoryError). Outliers keep FD unless they are
  extreme enough to hit that safety limit.
- `network_map`: repeated edges are summed instead of the last row winning;
  categorical node columns, non-string column labels and node order work;
  directed communities use summed reciprocal weights; weighted betweenness
  uses exact fractions so equally short paths share credit; directed graphs
  draw arrowheads (Matplotlib and Plotly).
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
  when no extension is given and accepts file-like objects.

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
