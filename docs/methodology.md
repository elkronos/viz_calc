# Methodology

This page lists every statistical and design decision in viz_calc, why it
was made, and where it comes from. The implementations are in
[`viz_calc.stats`](api.md#statistics) and are unit-tested against SciPy and
worked examples.

## Statistical methods

`pd.NA` in nullable `Int64`/`Float64` columns (or in a plain list passed to a
`viz_calc.stats` function) is treated exactly like `NaN`:
missing values are dropped (pairwise for correlations). Every `level=` is a
proportion strictly between 0 and 1 (`0.95`, not `95`); anything else raises
a `ValueError` instead of producing `NaN` intervals.

### Confidence interval for a mean
Student-*t* interval: $\bar x \pm t_{n-1,\,1-\alpha/2}\, s/\sqrt{n}$.
Used by `estimation_plot`, `benchmark_bar`.

### Welch's *t*-test (unequal variances)
$t = (\bar x_B - \bar x_A)/\sqrt{s_A^2/n_A + s_B^2/n_B}$ with
Welch–Satterthwaite degrees of freedom. The default because it keeps the
nominal error rate when variances or group sizes differ and costs almost no
power when they are equal (Welch, 1947; Delacre, Lakens & Leys, 2017).
If both groups are constant the standard error is zero and $t = \pm\infty$
with the sign of the difference ($p = 0$); two identical constants have no
difference, so $t$ is `NaN` and $p = 1$. Used by `estimation_plot`.

### Hedges' *g*
Cohen's *d* with pooled SD, times the small-sample correction
$J = 1 - 3/(4\,df - 1)$ (Hedges, 1981). CI from the large-sample variance
$\frac{n_A+n_B}{n_A n_B} + \frac{g^2}{2(n_A+n_B)}$ (Hedges & Olkin, 1985).
Undefined (`NaN`) when both groups are constant.

A group counts as constant when all its values are identical, so `[0.1] * 3`
behaves like `[1.0] * 3` even though its floating-point mean is off by an ulp:
its mean is taken to be its common value, so `[0.1] * 3` and `[0.1] * 10`
are identical groups and the mean CI of `[0.1] * 3` is exactly 0.1.
The mean CI, Welch's test and *g* rescale the data by a power of two before
squaring deviations; the rescaling is exact, so ordinary results are
unchanged and magnitudes beyond about $10^{\pm 154}$ no longer overflow or
underflow (which used to give $g = 0$).

### Bootstrap confidence intervals
Bias-corrected and accelerated (BCa) intervals (Efron, 1987) from
`scipy.stats.bootstrap`, 5,000 resamples, fixed seed. Each group is resampled
independently. Used for the mean difference in `estimation_plot`, following
estimation-statistics practice (Cumming, 2014; Ho et al., 2019).

### Wilson score interval for a proportion
$\dfrac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + z^2/n}$.
Recommended over the Wald interval, whose coverage collapses for small *n*
or extreme proportions (Wilson, 1927; Brown, Cai & DasGupta, 2001). Used by
`percent_grid`, `centered_bar`.

### Multiple-comparison adjustment
* **Holm** step-down (Holm, 1979): controls the family-wise error rate and
  is uniformly more powerful than Bonferroni. Default everywhere.
* **Benjamini–Hochberg** (1995): controls the false discovery rate. Better
  for screening many correlations.
* **Bonferroni** and **none** are available.

P-values outside $[0, 1]$ raise a `ValueError`.

Used by `estimation_plot` (across comparisons with the reference),
`correlogram` (across the *k(k−1)/2* unique pairs), `compare_correlations`
(within each pair of groups) and `profile_scatters`.

### Correlation with confidence interval
Pearson's *r* or Spearman's ρ with pairwise deletion of missing values.
The CI uses Fisher's $z = \operatorname{atanh}(r)$ with standard error
$1/\sqrt{n-3}$ (Fisher, 1921). For Spearman the standard error is
$\sqrt{1.06/(n-3)}$ (Fieller, Hartley & Pearson, 1957). The CI is `NaN` for
$n \le 3$ or $|r| = 1$ (up to rounding).

The Pearson p-value is the usual *t* test. For Spearman with $n \le 9$
complete pairs the p-value is exact: under independence all $n!$ pairings
of the ranks are equally likely, and p is the share of them whose $|\rho|$
is at least the observed one (tied values keep their midranks, so the test is
conditional on the ties). The *t* approximation is badly anti-conservative
there: it gives $p = 0$ for any perfect rank agreement, whereas the exact
two-sided p for $|\rho| = 1$ is $2/n!$, i.e. 0.33 at $n = 3$ and 0.083 at
$n = 4$, so no rank correlation of three or four pairs can reach $p < 0.05$.
From $n = 10$ the *t* approximation is used. `correlogram`,
`profile_scatters` and `quadrant_plot` inherit this.

### Difference between two independent correlations
$z = \dfrac{\operatorname{atanh} r_B - \operatorname{atanh} r_A}{\sqrt{SE_A^2 + SE_B^2}}$,
two-sided normal p-value (Fisher, 1921; Cohen, Cohen, West & Aiken, 2003).
Valid only when the groups contain different units; dependent correlations
need Steiger's (1980) test. The test is asymptotic, so read it with caution
for groups of only a few units; correlations of ±1 are clipped to ±0.999999.
Used by `compare_correlations`.

### Histogram bin width
Freedman–Diaconis by default: width $= 2\,\mathrm{IQR}\,n^{-1/3}$, which is
robust to outliers (Freedman & Diaconis, 1981). On heavily tied or
zero-inflated data the IQR can be zero (or zero up to a few ulps of
floating-point noise in the quartiles),
which would give one bin or billions; then, or if FD would need more than
100,000 bins, Sturges' rule is used instead and `info["bin_rule"]` records
why. Outliers keep FD unless they are extreme enough to hit that safety
limit. On whole-number data (counts, scores) an FD width below 1 leaves bins
that can hold no value (a comb of empty bars), and a fractional width makes
bins span different numbers of values (a sawtooth). There the width is
rounded to the nearest whole number (at least 1) and the edges sit on
half-integers from $\min(x) - 0.5$; `info["bin_rule"]` then reads
`fd (whole-number widths for integer data)`. The 100,000-bin limit is
checked on the bins built with the rounded width, since a width rounded
down (1.4 to 1) adds bins. Scott (1979) and
Sturges (1926) are also available and used as published. Bins are computed
once on all data so groups and facets are directly comparable.

### Kernel density estimates
Gaussian KDE with Scott's bandwidth rule (SciPy's default), adjustable with
`bw_method`. Raincloud densities are trimmed to each group's observed range.

### Largest-remainder apportionment
Hamilton's method: floor each exact quota, then give the remaining units
to the largest fractional remainders, so tile counts always sum to the grid
size (Balinski & Young, 1982). Used by `waffle`.

### Eta-squared
$\eta^2 = SS_\text{between}/SS_\text{total}$, the share of a numeric
variable's variance explained by a grouping. Used to rank panels in
`profile_boxes`.

### Principal component analysis
SVD of the centred (and by default standardized) data matrix; eigenvalues
$= s^2/(n-1)$; component signs fixed so the largest loading is positive
(Jolliffe & Cadima, 2016). Ellipses are drawn from the eigen-decomposition
of each group's covariance with radius $\sqrt{\chi^2_2(\text{level})}$. The
"data" ellipse uses the covariance of the points; the "confidence" ellipse
uses the covariance of the mean (divided by *n*).

### Networks
Repeated edges are merged by summing their weights. Louvain community
detection (Blondel et al., 2008) via NetworkX, seeded, on the undirected graph
(reciprocal directed weights are summed). Betweenness centrality (Brandes, 2001) uses
1/weight as edge length, so strong ties are short paths; zero-weight edges
are left out. Two paths of equal length must share the credit, but
floating-point rounding can make one of them look shorter, so how path
lengths are compared depends on whether the weights are stored exactly:

* **Exact weights.** Integers, and floats that cannot have been rounded
  when stored (they print exactly as stored, and are integers below
  2^(mantissa bits + 1), i.e. 2⁵³ for float64 and 2²⁴ for float32, or have no
  more digits than the type's precision: `2.0`, `0.125`, `1001.5`), are
  compared exactly. Larger float integers may be rounded counts and use the
  tolerant comparison. Floating point orders the paths; any two
  that come within 10⁻¹¹ of each other are then compared exactly, through
  their residues modulo the prime 2¹²⁷ − 1 and, if those differ, in rational
  arithmetic. Unlike rational arithmetic throughout, whose denominators grow
  along every path, this costs about as much as floating point.
* **Other floats** (`0.3`, `73/9`, shares such as count/438, float32 counts
  above 2²⁴) may stand for a value the type cannot hold, and which value was
  meant cannot be known (`0.98989898989899` may be a typed decimal or 98/99).
  Brandes' algorithm then treats path lengths within a relative 10⁻¹⁰ of the
  shortest as equal, the tolerance igraph uses (Csárdi & Nepusz, 2006),
  except that the slack never exceeds a quarter of the edge being added, so
  a real extra hop is not a tie (edges more than 10¹³ times shorter than the
  path, below the rounding of float64 sums, are the exception). float32 and
  float16 hold only about 7 and 3 significant digits; a value entered and
  then rescaled can be off by one machine epsilon, and two equal paths by
  two, so for them the tolerance is four epsilons (4.8 × 10⁻⁷ and 0.4%), and
  smaller differences cannot be resolved.

The choice is made for the whole weight column, so one inexact weight makes
every comparison tolerant. Either way the result does not depend on row
order. `info["betweenness_arithmetic"]` and `info["betweenness_tolerance"]`
record which comparison was used.

## Visual design decisions

| Decision | Reason | Source |
|---|---|---|
| Show raw data with summaries (raincloud, estimation plot, points on bars) | Bar-of-means charts hide *n*, spread and outliers | Weissgerber et al., 2015; Allen et al., 2019 |
| Default error bars are CIs, not ±1 SE | CIs have a stated coverage; SE bars are often misread | Cumming & Finch, 2005 |
| Dot plots and lengths on a common baseline for rankings | Position/length are judged more accurately than angle or area | Cleveland & McGill, 1984 |
| Okabe–Ito categorical palette | Distinguishable with common colour-vision deficiencies | Okabe & Ito, 2008 |
| Perceptually uniform (viridis) and diverging-centred colormaps | Rainbow maps distort data and exclude colour-blind readers | Crameri, Shephard & Heron, 2020 |
| Diverging stacked bars for Likert items | Aligns the neutral point so agreement reads as position | Robbins & Heiberger, 2011 |
| UpSet plots instead of Venn diagrams | Venn diagrams do not scale beyond three sets | Lex et al., 2014 |
| Bullet graphs in greyscale bands | Compact target comparison readable without colour | Few, 2013 |
| Bubble **area** ∝ value, one scale and fixed axes per animation | Area is the size cue readers use; per-frame rescaling breaks comparison between frames | Cleveland & McGill, 1984 |
| Label text colour chosen by contrast | Keeps labels legible on light and dark fills | WCAG 2.x contrast ratio |

## References

* Allen, M., Poggiali, D., Whitaker, K., Marshall, T. R., & Kievit, R. A. (2019). Raincloud plots: a multi-platform tool for robust data visualization. *Wellcome Open Research*, 4, 63.
* Balinski, M. L., & Young, H. P. (1982). *Fair Representation: Meeting the Ideal of One Man, One Vote*. Yale University Press.
* Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society B*, 57(1), 289–300.
* Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics*, P10008.
* Brandes, U. (2001). A faster algorithm for betweenness centrality. *Journal of Mathematical Sociology*, 25(2), 163–177.
* Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science*, 16(2), 101–133.
* Cleveland, W. S., & McGill, R. (1984). Graphical perception: theory, experimentation, and application to the development of graphical methods. *Journal of the American Statistical Association*, 79(387), 531–554.
* Cohen, J., Cohen, P., West, S. G., & Aiken, L. S. (2003). *Applied Multiple Regression/Correlation Analysis for the Behavioral Sciences* (3rd ed.). Erlbaum.
* Crameri, F., Shephard, G. E., & Heron, P. J. (2020). The misuse of colour in science communication. *Nature Communications*, 11, 5444.
* Csárdi, G., & Nepusz, T. (2006). The igraph software package for complex network research. *InterJournal, Complex Systems*, 1695.
* Cumming, G. (2014). The new statistics: why and how. *Psychological Science*, 25(1), 7–29.
* Cumming, G., & Finch, S. (2005). Inference by eye: confidence intervals and how to read pictures of data. *American Psychologist*, 60(2), 170–180.
* Delacre, M., Lakens, D., & Leys, C. (2017). Why psychologists should by default use Welch's t-test instead of Student's t-test. *International Review of Social Psychology*, 30(1), 92–101.
* Efron, B. (1987). Better bootstrap confidence intervals. *Journal of the American Statistical Association*, 82(397), 171–185.
* Few, S. (2013). *Bullet Graph Design Specification*. Perceptual Edge.
* Fieller, E. C., Hartley, H. O., & Pearson, E. S. (1957). Tests for rank correlation coefficients. I. *Biometrika*, 44(3/4), 470–481.
* Fisher, R. A. (1921). On the "probable error" of a coefficient of correlation deduced from a small sample. *Metron*, 1, 3–32.
* Freedman, D., & Diaconis, P. (1981). On the histogram as a density estimator: L2 theory. *Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete*, 57, 453–476.
* Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect size and related estimators. *Journal of Educational Statistics*, 6(2), 107–128.
* Hedges, L. V., & Olkin, I. (1985). *Statistical Methods for Meta-Analysis*. Academic Press.
* Ho, J., Tumkaya, T., Aryal, S., Choi, H., & Claridge-Chang, A. (2019). Moving beyond P values: data analysis with estimation graphics. *Nature Methods*, 16, 565–566.
* Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
* Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A*, 374, 20150202.
* Lex, A., Gehlenborg, N., Strobelt, H., Vuillemot, R., & Pfister, H. (2014). UpSet: visualization of intersecting sets. *IEEE Transactions on Visualization and Computer Graphics*, 20(12), 1983–1992.
* Okabe, M., & Ito, K. (2008). *Color Universal Design (CUD): How to make figures and presentations that are friendly to colorblind people*. J*Fly.
* Robbins, N. B., & Heiberger, R. M. (2011). Plotting Likert and other rating scales. *Proceedings of the 2011 Joint Statistical Meeting*, Section on Survey Research Methods, 1058–1066.
* Scott, D. W. (1979). On optimal and data-based histograms. *Biometrika*, 66(3), 605–610.
* Steiger, J. H. (1980). Tests for comparing elements of a correlation matrix. *Psychological Bulletin*, 87(2), 245–251.
* Sturges, H. A. (1926). The choice of a class interval. *Journal of the American Statistical Association*, 21(153), 65–66.
* Weissgerber, T. L., Milic, N. M., Winham, S. J., & Garovic, V. D. (2015). Beyond bar and line graphs: time for a new data presentation paradigm. *PLoS Biology*, 13(4), e1002128.
* Welch, B. L. (1947). The generalization of "Student's" problem when several different population variances are involved. *Biometrika*, 34(1–2), 28–35.
