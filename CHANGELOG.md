# Changelog

## Unreleased

### Changed
- Labels passed to a Matplotlib chart, such as `start_label=` in `dumbbell` or
  `labels=` in `timeseries_fill`, are drawn as written like the rest of the
  chart's text: a pair of `$` in them no longer starts mathtext. For mathtext in
  a chart's title or axis label, set it after the call
  (`res.axes.set_title(r"$\alpha$ = 0.05", parse_math=True)`). A dollar sign
  escaped as `\$`, in a label or in the data, now keeps its backslash
  (`\$5-\$10` used to draw `$5-$10`); drop the escape. A legend a chart adds to
  a caller's `ax=` (`dumbbell`, `likert`, `timeseries_fill`, ...) also shows the
  labels of the caller's own artists there as written.
- One column passed for two roles that cannot share it raises a clear error in
  the charts that still accepted it: `x` and `y` in `estimation_plot`,
  `benchmark_bar`, `lollipop`, `centered_bar` and `raincloud`; `x` and `hue` or
  `facet` in `histogram`; `x` and `group` in `ridgeplot`; `category` and `value`
  in `waffle`; `outer` or `inner` and `value` in `nested_pie`; `group` and
  `value` in `circular_bar`; `column` and `facet` in `percent_grid`; a band
  column named twice in `bullet`; a column listed in both `categorical` and
  `numeric` of `profile_boxes`; and a `pca_plot` `group` that is also one of its
  `features` (colour by a copy under another name instead). These calls drew one
  group per distinct value: `benchmark_bar(datasets.trial(), x="score",
  y="score")` drew 106 bars of 1 to 3 rows each, and `estimation_plot` failed
  with "groups need at least two observations". A label that is also the value
  (`waterfall`, `bullet`, `circular_bar`) or the group (`circular_bar`, and
  `task` in `gantt`), `bullet`'s value as its own target and `histogram`'s hue
  as its facet stay allowed.
- A numeric column holding `inf` or `-inf` raises a `ValueError` naming the
  column, before anything is drawn, in the charts for comparing groups,
  correlations, distributions and part-to-whole, in `profile_boxes`,
  `profile_scatters` and `animated_bubble`. Such values failed late with
  messages that did not name the column ("not enough values to unpack" in
  `lollipop`, `funnel` and `waterfall`, "Axis limits cannot be NaN or Inf",
  "Wedge sizes must be finite numbers", and with SciPy 1.11 "array must not
  contain infs or NaNs" from `correlogram` and `quadrant_plot`), or gave wrong
  results: `centered_bar(threshold="mean")` compared every group with an
  infinite mean, `circular_bar` drew 3 bars for a table of 4, `animated_bubble`
  gave every bubble zero area when one size was infinite, and on pandas 2.0
  `lollipop` reported a group's mean as NaN. Calls that coped with an infinite
  value raise too: `correlogram` and `compare_correlations` with
  `method="spearman"`, and `lollipop(stat="median")`. `lollipop(stat="count")`
  still counts infinite values, and the `viz_calc.stats` functions other than
  `wilson_ci` (see Fixed) handle them as before.

### Fixed
- `datasets.sales()` and `calendar_heatmap` read `DatetimeIndex.day_of_week`
  instead of `weekday`, which pandas 3.1 deprecates (`dayofweek` is deprecated
  as well). On pandas 3.1 the warning stopped the test suite at collection and
  showed in the pytest runs of anyone calling `calendar_heatmap` or
  `datasets.sales()`.
- `bootstrap_ci` and `estimation_plot` needed memory that grew with the square
  of the group size: 4 GB at 10,000 rows per group and over 7 GB at 14,000.
  When any group in a comparison has more than 2,000 rows, the resamples are
  now drawn in batches, which keeps peak memory near 400 MB. Those seeded
  intervals differ from 1.1.0 by Monte Carlo noise; when every group has 2,000
  rows or fewer, the resamples are drawn as in 1.1.0.
- The error for a missing optional dependency suggested
  `pip install "viz_calc[extra]"`. `viz_calc` is not on PyPI, so that command
  would install whatever someone else publishes under the name. The message
  now names the extra's packages and minimum versions.
- `pca_plot(ellipse="confidence")` sized the region for a group mean with the
  chi-square radius, which assumes a known covariance: a nominal 95% ellipse
  covered the true mean about 51% of the time in groups of 3, 75% in groups of 5
  and 87% in groups of 10. It now uses Hotelling's T² radius, √(2(n − 1)/(n − 2)
  · F₂,ₙ₋₂(level)), which covers at the stated level from 3 points on; the
  ellipse is about twice as wide at n = 5 and 11.5 times as wide at n = 3. A
  confidence ellipse no longer sets the axis limits, so a small group's ellipse
  is cut off at the edges of the plot rather than zooming the axes out until
  the points sit in its middle. The `"data"` ellipse keeps its chi-square
  radius and is now described as the normal-theory ellipse at the group's mean
  and covariance.
- The BCa intervals of `estimation_plot` and `stats.bootstrap_ci` changed with
  the units of the data: two groups holding the same values [1, 1, 1, 2, 1] and
  [1, 1, 1, 1, 2] gave −0.4 to 0.4 in units and −0.4 to 0.6 in tenths, because
  rounding noise in the mean difference moved bootstrap values that tie with the
  estimate to either side of it. With the default statistics, a bootstrap value
  closer to the estimate than 2⁻⁴⁰ times the data's largest magnitude now counts
  as a tie. Values far from zero next to their spread (a spread below 2⁻²⁰ of
  their magnitude, such as epoch times) are first shifted exactly by the value
  nearest zero, so the tolerance follows their spread; this also makes their
  intervals more accurate (by about 1e-6 of the interval width on continuous
  data). Other bootstraps without near ties, as on continuous data, are
  bit-identical to 1.1.0. On data recorded to a fixed precision a seeded
  interval moves by a fraction of a bootstrap step where a tie flipped: on
  `datasets.trial()` the control vs low dose interval goes from −2.216 to 8.835
  to −2.218 to 8.835, and the one-decimal table in the comparing-groups
  walkthrough is unchanged. A statistic passed to `bootstrap_ci` is used as it
  is.
- `stats.bootstrap_ci(x)` with one sample failed with an internal `TypeError`
  from the two-sample default statistic; it now bootstraps the mean. Three or
  more samples without `statistic=` raise a `ValueError` that names the count.
  `method`, `n_resamples` and `seed` are checked before any resampling:
  `method=None` raised `AttributeError`, `n_resamples=True` returned `nan`
  limits, and constant samples accepted any option. `n_resamples` must be an
  `int`, as in `estimation_plot` (a float such as `200.0` now raises and names
  the `int` to use), and `seed` must be `None`, an `int` of at least 0 or a
  NumPy `Generator`, `BitGenerator` or `SeedSequence` (a list of ints is no
  longer passed on to NumPy). `method` still accepts any letter case.
- `bootstrap_ci` chose SciPy's `rng` or `random_state` keyword by retrying on
  `TypeError`, so on SciPy 1.15 and later a `TypeError` raised by the caller's
  statistic reran the whole bootstrap and was chained to itself, and on older
  SciPy the real error carried an unrelated "unexpected keyword" context. The
  keyword is now read once from SciPy's signature.
- `stats.wilson_ci` returned `(nan, nan)` for 5 successes of 3, a negative count
  or a negative n, `(0, 0)` for an infinite n, and a finite interval for 10.5
  successes of 10. Impossible counts now raise a `ValueError` that names them. A
  missing count or n = 0 still gives `nan`, fractional (weighted) counts are
  still accepted, and counts near 1e200 no longer warn about overflow.
- `pca` and `pca_plot` standardized a constant feature whose value is not a
  binary fraction (0.1, 1/3) into a component holding 20% of the variance,
  because its floating-point SD is about 3e-17 instead of 0. They now raise the
  same "constant feature(s) cannot be scaled" error as for a constant 0.5, and
  with `scale=False` such a feature gets exactly zero variance.
- `pca` and `pca_plot` raised `NotImplementedError` for Arrow-backed numeric
  features (`dtype_backend="pyarrow"`) on pandas 2.0.
- `profile_boxes` gave a constant numeric column such as 0.1 an eta-squared of 1
  or 0 from rounding noise, a different pair on each pandas pin, so it could
  head the table. Its eta-squared is now missing, as for a constant 0.5.
- `welch_test`, `hedges_g`, Pearson's `correlation_test`, the eta-squared of
  `profile_boxes` and `pca` lost the values' digits when the values sit far from
  zero next to their spread: with an exact offset of 2⁵³, Welch's *t* was 2.449
  instead of 2.484, *r* of an exact linear relation 0.985 (with a SciPy
  near-constant warning) and the eta-squared of groups [0, 2] and [4, 8] 1.11
  instead of 0.71; an offset of 2⁵² on one PCA feature moved the explained
  variance ratios by 1.7e-4. When the spread is below 2⁻²⁰ of the values'
  magnitude (per feature for `pca`), the value nearest zero is now subtracted
  first, which is exact there. Other data are not shifted and give bit-identical
  results.
- `histogram` reported `bin_rule` `"fd"` (or `"auto"`) for a column holding a
  single value, although a zero IQR is documented to fall back to Sturges and
  say why; it now reads `"sturges (IQR is 0, so FD is undefined)"`. The bins
  were already the Sturges ones.
- `histogram(alpha=)`, `ridgeplot(overlap=)` and `to_pptx(dpi=)` accepted a
  `Decimal`, `Fraction` or 0-d array in their range check and then failed inside
  Matplotlib or in arithmetic with a `TypeError`. They now use the checked
  float.
- On pandas 2.2 and 2.3, `gantt`, `duration_plot` and `timeseries_fill` accepted
  a text column whose values shared one UTC offset next to a value without one,
  and silently gave that value the shared offset (or read it as UTC next to `Z`
  values). The documented "mixes values with and without a UTC offset" error now
  fires on every pandas version, for `Z` and hour-only offsets (`-04`) too. The
  check reads each distinct value once, which adds about 1 s per million text
  values with offsets.
- Arrow-backed date columns, which `pd.read_parquet` and
  `pd.read_csv(engine="pyarrow")` return with `dtype_backend="pyarrow"` and
  `convert_dtypes(dtype_backend="pyarrow")` gives, failed in `period_bars`,
  `timeseries_fill`, `calendar_heatmap`, `gantt` and `duration_plot`: on pandas
  3.1 with `NotImplementedError` (`date32`) or `AttributeError` (`period_bars`
  on Arrow timestamps), and on pandas 2.1 to 2.3 with "could not be parsed as
  dates" for Arrow timestamps. They now give the same tables as NumPy datetimes,
  keep their time zone, and the tables hold NumPy datetime columns.
- On pandas 2.0 to 2.3, `gantt` and `circular_bar` matched no value of an
  Arrow-backed timestamp `group` column, with or without a time zone, and
  `nested_pie` none of such an `outer` column; on pandas 2.2 the same happened
  to Arrow-backed durations. `gantt` drew every bar grey as "(missing)" while
  its legend listed the groups, and `circular_bar` and `nested_pie` raised a
  `ValueError` that did not name the column. They now match such a column by
  its values.
- `calendar_heatmap` drew a row for every year from the earliest to the latest
  date, so one placeholder date such as `9999-12-31` next to a year of data
  built 7,976 empty panels and took over two minutes and 4.5 GB on pandas 3.1;
  `1900-01-01` gave 125 rows. Only years that hold data get a row now, so a gap
  year in ordinary data is left out rather than drawn as an all-grey row.
- `timeseries_fill` with time-zone-aware times took 29 s for a million hourly
  rows (naive times: 2 s), because Matplotlib converted every Timestamp in
  Python. It now draws UTC date numbers on an axis labelled in the data's zone,
  with the same ticks, limits and pixels: 6 s on pandas 3.1, 7 s on pandas 2.0
  (was 33 s). ISO text with mixed UTC offsets, as any series across a
  daylight-saving change has, was parsed one value at a time; a million such
  values now take 3.6 s instead of 10.4 s in `period_bars`, 4.9 s instead of
  11.3 s in `calendar_heatmap` and 4.2 s instead of 27.2 s in `timeseries_fill`
  (pandas 3.1; on pandas 2.0, 13 s instead of 26 s, 15 s instead of 28 s and 15
  s instead of 46 s). Tables are unchanged; for time-zone-aware times the drawn
  lines now hold Matplotlib date numbers instead of Timestamps.
- Category values, column names and labels holding a pair of `$` were drawn as
  Matplotlib mathtext in all 35 Matplotlib charts. A price band such as `$5-$10`
  lost both dollar signs, and text that is not valid mathtext, such as `$a^$`,
  raised a `ValueError`: during the call in the charts that fit their layout
  around the text (`histogram`, `percent_grid`, `donut_grid`, `bullet`,
  `estimation_plot`, `compare_correlations`, `animated_bubble` and the
  `profile_*` functions), and otherwise only later, when the figure was saved
  (`waffle`, `lollipop`, `upset`, `network_map`, ...). Text a chart draws is now
  drawn as written, also on a caller's `ax=`, in tick labels Matplotlib makes
  when the figure is saved and in `animated_bubble`'s later frames.
  `text.parse_math` is off only while a call runs; text already on the caller's
  Axes, text added or changed afterwards and the tick labels of a log axis keep
  Matplotlib's mathtext.
- `sankey` and `network_map(interactive=True)` passed node names to Plotly as
  they were, and Plotly reads a subset of HTML in labels and hover text: a node
  named `<a href="https://...">Click</a>` became a live link in a saved HTML
  file, `A<br>B` was split over two lines and `AT&amp;T` showed as `AT&T`. Node
  names are now escaped and show as written; the hover text keeps its bold name
  and line breaks, and `sankey`'s `title=` is passed through unchanged.
- `centered_bar`, `likert` and `stacked_percentages` results could not be
  pickled, for a cache or a process pool: Matplotlib places centred bar labels
  with a lambda, and `centered_bar` and `likert` formatted their ticks with
  lambdas (so `labels=False` did not help). All three pickle now and draw the
  same pixels as before. `animated_bubble`'s `info["animation"]` still cannot be
  pickled, because a Matplotlib `FuncAnimation` never can; its figure and table
  can.
- The value labels of `lollipop`, `dumbbell(show_values=True)`, `waterfall`
  and `funnel` raised "not enough values to unpack" for an infinite value, such
  as a `lollipop(stat="sum")` or a `waterfall` total that overflows; they now
  read `inf` and `-inf`. Below 1 they wrote a subnormal such as 5e-324 with six
  significant digits (`4.94066e-324`) instead of the documented two
  (`4.9e-324`).
- On pandas 2.0 and 2.1 (viz_calc declares `pandas>=2.0`, and CI tests 2.0.3),
  `estimation_plot`, `benchmark_bar`, `centered_bar`, `percent_grid`,
  `histogram` with `hue` or `facet`, `ridgeplot`, `raincloud`,
  `compare_correlations` and, with copy-on-write on, `profile_boxes` failed with
  "buffer source array is read-only" when a text column was backed by a
  read-only array, as in a frame built with `copy=False` over one. pandas 2.2
  and later were not affected.
- `animated_bubble` failed when the figure was rendered (`savefig`, `to_jshtml`,
  notebook display or `anim.save`, after the call itself had returned) with
  "need at least one array to concatenate" when every bubble of a colour group
  in a frame, or every bubble of a frame without `color`, had no size. A bubble
  without x, y or size is now left out of its frame, and every colour group
  keeps its legend entry in every frame.
- The `value` column of the `waffle` and `nested_pie` tables is int64 or float64
  for nullable (`Int64`, `Float64`, `boolean`) input. On pandas 2.0 and 2.1 it
  was object for `waffle`, so `select_dtypes("number")` dropped it, and on every
  pandas version `nested_pie` kept the nullable dtype.

### Documentation
- The `to_pptx` docstring and the walkthroughs point to the install section
  instead of `pip install "viz_calc[...]"`.
- The `welch_test` docstring and the methodology page say that two constant
  groups get the Student degrees of freedom n_a + n_b − 2, since the
  Welch–Satterthwaite formula is 0/0 there; `estimation_plot` shows the same
  value in its `df` column.
- The methodology page describes the tie rule of the bootstrap, both PCA
  ellipses (citing Hotelling, 1931), the domain of the Wilson interval, the
  exact shift for values far from zero and the missing eta-squared of a constant
  column. The PCA walkthrough describes the two ellipses the same way as the
  `pca_plot` docstring.
- The time walkthrough lists the date columns the time charts accept,
  Arrow-backed ones included, and says that `timeseries_fill`, `gantt` and
  `duration_plot` reject text that mixes values with and without a UTC offset.
  It, the `calendar_heatmap` docstring and the getting-started table say that
  only years with data get a row.
- Getting started has a "Text in charts" section on how text from the data is
  drawn, in Matplotlib and in the Plotly figures, and what keeps mathtext.
- The `VizResult` docstring says which results cannot be pickled:
  `animated_bubble`'s animation, and the figure of `network_map(directed=True)`
  with networkx older than 3.3 (the CI minimum is 3.2.1), whose arrows networkx
  draws with a local function.
- Getting started lists, under Conventions, which column roles must be different
  columns and which may share one, and which charts reject infinite values.

### Development
- CI also runs every Monday against the newest dependency releases, and can be
  started by hand.
- The plotting contract tests build their datasets on first use, so a warning
  from one dataset fails the tests that use it instead of stopping collection
  of the whole file.
- A separate Attribution workflow fails a pull request or push to `main` when a
  commit message, an added line, the PR title, the PR description or the
  branch name carries an attribution trailer or names a writing tool. Its log
  names where each match is and omits the matched text.
- Tests read the same dates through `read_csv(engine="pyarrow")` and
  `convert_dtypes(dtype_backend="pyarrow")`, naive and time-zone-aware, and
  compare the tables of `period_bars`, `timeseries_fill`, `calendar_heatmap`,
  `gantt` and `duration_plot` with the NumPy result. They are skipped when
  pyarrow is not installed, as in every CI job on `main`.
- The three copies of the overflow-safe sum (`composition._sum`,
  `timeseries._sum_by` and `_core.safe_sum`) are one: `safe_sum` takes a list of
  keys and `dropna=`, and `_sum_by` calls it.

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
- Many options and counts are checked before anything is drawn: a bad
  option raises a `ValueError` listing the valid ones, and a bad count an
  error saying what is allowed. Among them: `lollipop(sort=...)` (`"asc"`
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
- An all-missing column that a function needs values from raises a clear
  error (in most functions "column ... has no non-missing values") instead of
  an internal error or an empty chart; one column passed for two roles raises
  a clear error in `dumbbell`, `divergent_bar`, `quadrant_plot`,
  `stacked_percentages`, `nested_pie` (outer and inner), `funnel`,
  `compare_correlations` (group among the columns), `sankey`, `network_map`,
  `gantt`, `duration_plot` and `animated_bubble` (for example
  `dumbbell(start="a", end="a")`).
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
  tolerance of 1e-10 (four machine epsilons for float32/float16), with the
  slack capped at a quarter of a path's last edge; results do not depend on
  row order
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
  selected as columns, not read as a row mask (pandas itself cannot tell `0`
  from `False`, or `1` from `True`, in the same frame's labels);
  `correlogram`, `compare_correlations`, `donut_grid`, `likert`, `pca`,
  `pca_plot`, `radar`, `timeseries_fill` and the `profile_*` functions reject
  a repeated column; `likert` rejects response levels named `item`, `n` or
  `net`.
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
- `likert` rejects a repeated response level and `donut_grid` a repeated
  column (both gave wrong numbers); `alpha`, `decimals`, `components`,
  `loadings`, `n_resamples`, `overlap`, `max_levels`/`min_levels`, `dpi`,
  `min_label` and `inner_radius` are checked; `gantt`, `duration_plot` and
  `timeseries_fill` reject columns named like their output columns.
- `divergent_bar` works on nullable columns with missing values;
  `divergent_bar`, `lollipop`, `calendar_heatmap` and `nested_pie` sum in
  float where an int64 sum would wrap, and `dumbbell` gives a float change
  instead of a wrapped int64; thresholds print with enough digits to agree
  with the classification; `histogram`'s density curve keeps the bars' scale
  when explicit bins leave values outside.
- `animated_bubble` orders day-first date labels by date, rejects labels
  that fit both day-first and month-first with different orders, and treats
  year-less labels ("1/2") as text on every pandas version.
- `network_map` rescales very small weights for communities and layout (it
  raised `ZeroDivisionError`) and rejects complex weights; `sankey` rejects
  infinite and complex flows.
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
