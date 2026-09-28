# Changelog

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
