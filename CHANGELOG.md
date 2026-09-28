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
- Unknown correlation methods, a `success` value that never occurs, empty
  data, all-missing grouping columns, one-sided quadrant splits, missing edge
  weights and non-positive `upset(max_intersections=...)` now raise clear errors.
- Number labels keep significant digits below 1 (`0.034`, not `0`); axes that
  reach 10,000 use K/M/B/T with precision chosen from the tick spacing.

### Fixed
- Histograms fall back from Freedman–Diaconis to Sturges when the IQR is zero
  or near zero (previously one bin, or a MemoryError).
- `network_map`: repeated edges are summed instead of the last row winning;
  categorical node columns, non-string column labels and node order work;
  directed communities use summed reciprocal weights.
- Dates: timezone-aware values (including mixed UTC offsets across DST) are
  placed by local time; mixed ISO date/date-time strings parse; string dates
  sort chronologically in `timeseries_fill`; numeric years stored as objects
  stay numeric.
- `calendar_heatmap` shows all-missing days as no data, not zero.
- `upset` no longer counts missing values or `"0"` strings as membership.
- `radar` leaves a missing group metric as a gap (with vertex markers) and
  handles nullable dtypes; `pca_plot` keeps categorical group order with any
  index; `correlogram(hide_nonsignificant=True)` keeps `info["matrix"]`.
- `raincloud` draws densities for narrow groups; `bullet` shows negative
  measures and accepts array bands; `waterfall` uses `start_label`;
  `gantt`/`animated_bubble` draw missing groups in grey with a legend entry;
  `animated_bubble` handles NaN and nullable columns; `dumbbell` handles
  booleans without casting integers; `donut_grid` skips boolean columns.
- `to_pptx` keeps each image's true aspect ratio and puts one title per item
  on every page of multi-page results; `VizResult.save` returns the real path
  when no extension is given.

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
