"""Statistical building blocks used by the plots.

Each function is a small, tested implementation of a published method, so the
numbers a chart shows can be traced back to a citation. They are public: use
them on their own when you want the numbers without the picture.

References are listed in each docstring and collected on the *Methodology*
page of the documentation.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Sequence
from functools import lru_cache
from itertools import permutations
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats as _st

__all__ = [
    "wilson_ci",
    "mean_ci",
    "welch_test",
    "hedges_g",
    "bootstrap_ci",
    "adjust_pvalues",
    "correlation_test",
    "compare_correlations_test",
    "histogram_bins",
    "largest_remainder",
]


def _as_float(x: Any) -> np.ndarray:
    """*x* as a float array, with missing values (``None``, ``nan``, ``pd.NA``) as ``nan``."""
    if isinstance(x, (pd.Series, pd.Index, pd.DataFrame, pd.api.extensions.ExtensionArray)):
        # np.asarray(x, dtype=float) raises for nullable dtypes holding pd.NA on pandas 2.0
        return x.to_numpy(dtype=float, na_value=np.nan)
    return np.asarray(x, dtype=float)


def _clean(x: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    arr = _as_float(x).ravel()
    return arr[~np.isnan(arr)]


def _check_level(level: float) -> None:
    """Raise unless *level* is a confidence level strictly between 0 and 1 (``0.95``, not ``95``)."""
    if not (isinstance(level, numbers.Real) and 0 < level < 1):
        raise ValueError(f"level must be strictly between 0 and 1 (e.g. 0.95 for a 95% interval), got {level!r}")


def _unit_scale(*samples: np.ndarray) -> float:
    """A power of two close to the largest magnitude in *samples* (1 if there is none).

    Dividing by a power of two is exact, so results for ordinary data are unchanged,
    but squared deviations can no longer overflow (``|x|`` above about 1e154) or
    underflow. *t*, *g* and *df* do not depend on the scale; differences and
    standard errors are multiplied back by it.
    """
    top = max((float(np.max(np.abs(s))) for s in samples if s.size), default=0.0)
    if top == 0 or not np.isfinite(top):
        return 1.0
    return float(np.ldexp(1.0, np.frexp(top)[1] - 1))


def _var(x: np.ndarray) -> float:
    """Sample variance (``ddof=1``), exactly 0 for a constant sample.

    The floating-point mean of a constant sample can be off by an ulp
    (``mean([0.1] * 3) != 0.1``), which would leave a spurious variance of about
    1e-34 and turn a zero-variance case into an astronomically large *t* or *g*.
    """
    return 0.0 if np.ptp(x) == 0 else float(x.var(ddof=1))


def wilson_ci(successes: int | np.ndarray, n: int | np.ndarray, level: float = 0.95) -> tuple[Any, Any]:
    """Wilson score interval for a binomial proportion.

    Preferred over the textbook Wald interval, which has poor coverage for
    small *n* or proportions near 0 or 1 (Brown, Cai & DasGupta, 2001).

    Returns ``(lower, upper)`` as proportions: floats for scalar input,
    arrays for array input. ``n == 0`` gives ``nan``. *level* must be
    strictly between 0 and 1 (``0.95``, not ``95``), else ``ValueError``.

    References
    ----------
    Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. *JASA*, 22(158), 209–212.
    Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a
    binomial proportion. *Statistical Science*, 16(2), 101–133.
    """
    _check_level(level)
    k = _as_float(successes)
    n = _as_float(n)
    z = _st.norm.ppf(1 - (1 - level) / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lower = np.where(n > 0, np.clip(centre - half, 0, 1), np.nan)
    upper = np.where(n > 0, np.clip(centre + half, 0, 1), np.nan)
    lower = np.where((n > 0) & (k == 0), 0.0, lower)  # exact at the boundaries (rounding can leave 1e-17)
    upper = np.where((n > 0) & (k == n), 1.0, upper)
    if lower.ndim == 0:
        return float(lower), float(upper)
    return lower, upper


def mean_ci(x: Sequence[float], level: float = 0.95) -> tuple[float, float, float]:
    """Mean and its t-based confidence interval: ``(mean, lower, upper)``.

    Uses the Student *t* distribution with ``n - 1`` degrees of freedom. With
    fewer than two observations the interval is ``nan``. *level* must be
    strictly between 0 and 1 (``0.95``, not ``95``), else ``ValueError``.
    """
    _check_level(level)
    x = _clean(x)
    n = x.size
    if n == 0:
        return (np.nan, np.nan, np.nan)
    s = _unit_scale(x)
    x = x / s  # exact; see _unit_scale
    m = float(x.mean() * s)
    if n < 2:
        return (m, np.nan, np.nan)
    se = np.sqrt(_var(x)) / np.sqrt(n) * s
    t = _st.t.ppf(1 - (1 - level) / 2, n - 1)
    return (m, m - t * se, m + t * se)


def welch_test(a: Sequence[float], b: Sequence[float], level: float = 0.95) -> dict[str, float]:
    """Welch's unequal-variance *t*-test for ``mean(b) - mean(a)``.

    Welch's test keeps its error rate when group variances or sizes differ and
    loses almost nothing when they are equal, so it is the recommended default
    over Student's *t*-test (Delacre, Lakens & Leys, 2017).

    Returns a dict with ``difference``, ``ci_low``, ``ci_high``, ``t``,
    ``df`` (Welch–Satterthwaite) and ``p``. When both groups are constant the
    standard error is zero: ``t`` is then ``±inf`` with the sign of the
    difference (``nan`` if there is none) and ``p`` is 0 (1).

    References
    ----------
    Welch, B. L. (1947). The generalization of 'Student's' problem when several
    different population variances are involved. *Biometrika*, 34, 28–35.
    Delacre, M., Lakens, D., & Leys, C. (2017). Why psychologists should by
    default use Welch's t-test instead of Student's t-test. *International
    Review of Social Psychology*, 30(1), 92–101.
    """
    _check_level(level)
    a, b = _clean(a), _clean(b)
    if a.size < 2 or b.size < 2:
        raise ValueError("each group needs at least two non-missing values")
    s = _unit_scale(a, b)
    a, b = a / s, b / s  # exact; see _unit_scale
    va, vb = _var(a) / a.size, _var(b) / b.size
    diff = b.mean() - a.mean()
    se = np.sqrt(va + vb)
    if se == 0:
        return {"difference": float(diff * s), "ci_low": float(diff * s), "ci_high": float(diff * s),
                "t": float(np.copysign(np.inf, diff)) if diff else np.nan, "df": float(a.size + b.size - 2),
                "p": 0.0 if diff else 1.0}
    df = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    t = diff / se
    p = 2 * _st.t.sf(abs(t), df)
    crit = _st.t.ppf(1 - (1 - level) / 2, df)
    return {"difference": float(diff * s), "ci_low": float((diff - crit * se) * s),
            "ci_high": float((diff + crit * se) * s), "t": float(t), "df": float(df), "p": float(p)}


def hedges_g(a: Sequence[float], b: Sequence[float], level: float = 0.95) -> dict[str, float]:
    """Hedges' *g* standardized mean difference for ``b - a`` with a CI.

    Cohen's *d* with the pooled SD, multiplied by the small-sample correction
    ``J = 1 - 3 / (4 * df - 1)``. The interval uses the large-sample variance
    from Hedges & Olkin (1985). When both groups are constant the pooled SD is
    zero and *g* is undefined: every value is ``nan``.

    References
    ----------
    Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect
    size and related estimators. *Journal of Educational Statistics*, 6(2),
    107–128.
    Hedges, L. V., & Olkin, I. (1985). *Statistical Methods for
    Meta-Analysis*. Academic Press.
    """
    _check_level(level)
    a, b = _clean(a), _clean(b)
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        raise ValueError("each group needs at least two non-missing values")
    s = _unit_scale(a, b)
    a, b = a / s, b / s  # exact, and g does not depend on the scale; see _unit_scale
    df = n1 + n2 - 2
    sp = np.sqrt(((n1 - 1) * _var(a) + (n2 - 1) * _var(b)) / df)
    if sp == 0:
        return {"g": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    j = 1 - 3 / (4 * df - 1)
    g = j * (b.mean() - a.mean()) / sp
    se = np.sqrt((n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2)))
    z = _st.norm.ppf(1 - (1 - level) / 2)
    return {"g": float(g), "ci_low": float(g - z * se), "ci_high": float(g + z * se)}


def bootstrap_ci(
    *samples: Sequence[float],
    statistic: Callable[..., float] = lambda a, b: np.mean(b) - np.mean(a),
    level: float = 0.95,
    n_resamples: int = 5000,
    method: Literal["BCa", "percentile", "basic"] = "BCa",
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """Bootstrap estimate and confidence interval: ``(estimate, lower, upper)``.

    Defaults to the bias-corrected and accelerated (BCa) interval (Efron,
    1987), resampling each sample independently. The default statistic is the
    difference in means ``mean(b) - mean(a)`` used by estimation plots
    (Ho et al., 2019). A fixed *seed* keeps figures reproducible. *level* must
    be strictly between 0 and 1 (``0.95``, not ``95``), else ``ValueError``.

    References
    ----------
    Efron, B. (1987). Better bootstrap confidence intervals. *JASA*, 82(397),
    171–185.
    Ho, J., Tumkaya, T., Aryal, S., Choi, H., & Claridge-Chang, A. (2019).
    Moving beyond P values: data analysis with estimation graphics. *Nature
    Methods*, 16, 565–566.
    """
    estimate, lo, hi, _ = _bootstrap(samples, statistic, level, n_resamples, method, seed)
    return estimate, lo, hi


def _bootstrap(samples, statistic, level, n_resamples, method, seed):
    """Like :func:`bootstrap_ci` but also returns the bootstrap distribution."""
    _check_level(level)
    data = tuple(_clean(s) for s in samples)
    if any(d.size < 2 for d in data):
        raise ValueError("each sample needs at least two non-missing values")
    estimate = float(statistic(*data))
    kwargs = dict(n_resamples=n_resamples, confidence_level=level, method=method, vectorized=False)
    try:
        res = _st.bootstrap(data, statistic, rng=np.random.default_rng(seed), **kwargs)
    except TypeError:  # SciPy < 1.15 names the argument random_state
        res = _st.bootstrap(data, statistic, random_state=np.random.default_rng(seed), **kwargs)
    lo, hi = res.confidence_interval
    return estimate, float(lo), float(hi), np.asarray(res.bootstrap_distribution)


def adjust_pvalues(p: Sequence[float], method: Literal["holm", "fdr_bh", "bonferroni", "none"] = "holm") -> np.ndarray:
    """Adjust p-values for multiple comparisons. ``nan`` values are ignored.

    * ``"holm"`` – Holm (1979) step-down; controls the family-wise error rate
      and is uniformly more powerful than Bonferroni.
    * ``"fdr_bh"`` – Benjamini & Hochberg (1995); controls the false discovery
      rate, suited to screening many correlations.
    * ``"bonferroni"`` – multiply by the number of tests.
    * ``"none"`` – return the input unchanged.

    References
    ----------
    Holm, S. (1979). A simple sequentially rejective multiple test procedure.
    *Scandinavian Journal of Statistics*, 6(2), 65–70.
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
    rate. *JRSS B*, 57(1), 289–300.
    """
    methods = ("holm", "fdr_bh", "bonferroni", "none")
    if method not in methods:  # checked first, so an all-missing input cannot hide a typo
        raise ValueError(f"method must be one of {list(methods)}, got {method!r}")
    p = _as_float(p)
    out = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    q = p[mask]
    m = q.size
    if m == 0 or method == "none":
        out[mask] = q
        return out
    order = np.argsort(q)
    ranked = q[order]
    if method == "bonferroni":
        adj = np.minimum(q * m, 1.0)
    elif method == "holm":
        stepped = np.maximum.accumulate((m - np.arange(m)) * ranked)
        adj = np.empty(m)
        adj[order] = np.minimum(stepped, 1.0)
    else:  # fdr_bh
        stepped = np.minimum.accumulate((m / np.arange(m, 0, -1)) * ranked[::-1])[::-1]
        adj = np.empty(m)
        adj[order] = np.minimum(stepped, 1.0)
    out[mask] = adj
    return out


def _fisher_se(n: np.ndarray | float, method: str) -> np.ndarray | float:
    # Spearman's rho has a larger sampling variance than Pearson's r; the
    # 1.06 / (n - 3) approximation is from Fieller, Hartley & Pearson (1957).
    # The standard error is undefined for n <= 3; return nan rather than inf/0.
    n = _as_float(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 3, np.sqrt((1.06 if method == "spearman" else 1.0) / (n - 3)), np.nan)


def _check_method(method: str) -> None:
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be one of ['pearson', 'spearman'], got {method!r}")


# Largest n for which Spearman's p-value comes from the exact permutation distribution
# (9! = 362,880 pairings); above it the t approximation is used.
_SPEARMAN_EXACT_MAX_N = 9


@lru_cache(maxsize=16)
def _permutations(n: int) -> np.ndarray:
    """All ``n!`` orderings of ``range(n)`` as an ``(n!, n)`` array."""
    return np.array(list(permutations(range(n))), dtype=np.int8)


@lru_cache(maxsize=256)
def _spearman_null(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Exact null distribution of ``|T|`` for doubled midranks *a* and *b* (see :func:`_spearman_exact_p`).

    Returns the sorted distinct values of ``|T|`` over all ``n!`` pairings and, for
    each, how many pairings reach at least that value. It depends only on the two
    multisets of ranks, so it is cached: data without ties share one table per *n*.
    """
    av, bv = np.array(a, dtype=float), np.array(b, dtype=float)  # small integers: float arithmetic is exact
    t = np.abs(av.size * (bv[_permutations(av.size)] @ av) - av.sum() * bv.sum())
    values, counts = np.unique(t, return_counts=True)
    return values, np.cumsum(counts[::-1])[::-1]


def _spearman_exact_p(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided p-value of Spearman's rho from its exact permutation distribution.

    Under independence every pairing of the *y* ranks with the *x* ranks is
    equally likely, so p is the share of the ``n!`` pairings whose ``|rho|`` is
    at least the observed one. Tied values keep their midranks (the test is then
    conditional on the ties). Doubled midranks are integers and rho is
    proportional to ``T = n·Σ a_i b_i − Σa·Σb``, so the comparison is exact.
    """
    a = np.rint(2 * _st.rankdata(x)).astype(int)
    b = np.rint(2 * _st.rankdata(y)).astype(int)
    t_obs = abs(a.size * float(a @ b) - float(a.sum()) * float(b.sum()))
    values, at_least = _spearman_null(tuple(sorted(a.tolist())), tuple(sorted(b.tolist())))
    return float(at_least[np.searchsorted(values, t_obs)] / at_least[0])


def correlation_test(x: Sequence[float], y: Sequence[float], method: Literal["pearson", "spearman"] = "pearson",
                     level: float = 0.95) -> dict[str, float]:
    """Correlation with p-value and Fisher-*z* confidence interval.

    Pairs with a missing value in either variable are dropped. Returns
    ``r``, ``ci_low``, ``ci_high``, ``p`` and ``n``.

    The Pearson p-value is the usual *t* test. For Spearman with at most 9
    complete pairs the p-value is exact: the share of all ``n!`` pairings of
    the ranks whose ``|rho|`` is at least the observed one (ties keep their
    midranks). The *t* approximation that SciPy uses is far too liberal there;
    it gives ``p = 0`` for any perfect rank agreement, while the exact
    two-sided p for ``|rho| = 1`` is ``2/n!`` (0.33 at n = 3, 0.083 at
    n = 4). Larger samples use the *t* approximation. The CI needs ``n > 3``
    and ``|r| < 1`` (up to rounding); otherwise it is ``nan``. *level* must be
    strictly between 0 and 1 (``0.95``, not ``95``), else ``ValueError``.

    References
    ----------
    Fisher, R. A. (1921). On the "probable error" of a coefficient of
    correlation deduced from a small sample. *Metron*, 1, 3–32.
    Fieller, E. C., Hartley, H. O., & Pearson, E. S. (1957). Tests for rank
    correlation coefficients. I. *Biometrika*, 44(3/4), 470–481.
    """
    _check_method(method)
    _check_level(level)
    x = _as_float(x)
    y = _as_float(y)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    n = x.size
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"r": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": n}
    res = _st.spearmanr(x, y) if method == "spearman" else _st.pearsonr(x, y)
    r, p = float(res[0]), float(res[1])
    if method == "spearman" and n <= _SPEARMAN_EXACT_MAX_N:
        p = _spearman_exact_p(x, y)
    # |r| = 1 makes atanh infinite; rounding can leave a perfect correlation at 0.9999999999999999.
    if n > 3 and abs(r) < 1 - 8 * np.finfo(float).eps:
        z = np.arctanh(r)
        half = _st.norm.ppf(1 - (1 - level) / 2) * _fisher_se(n, method)
        lo, hi = float(np.tanh(z - half)), float(np.tanh(z + half))
    else:
        lo = hi = np.nan
    return {"r": r, "ci_low": lo, "ci_high": hi, "p": p, "n": n}


def compare_correlations_test(r1: float | np.ndarray, n1: int | np.ndarray, r2: float | np.ndarray,
                              n2: int | np.ndarray, method: Literal["pearson", "spearman"] = "pearson"
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Test whether two *independent* correlations differ: ``(z, p)``.

    Fisher's r-to-z test: ``z = (atanh r1 - atanh r2) / sqrt(se1² + se2²)``,
    two-sided. Works element-wise on arrays, e.g. two correlation matrices.
    Only valid when the correlations come from different people/units. The
    result is ``nan`` wherever either ``n`` is 3 or less. The test is
    asymptotic, so read it with caution for groups of only a few units; a
    correlation of ±1 is clipped to ±0.999999, which gives a very small
    p-value even for tiny groups.

    References
    ----------
    Fisher, R. A. (1921). *Metron*, 1, 3–32.
    Cohen, J., Cohen, P., West, S. G., & Aiken, L. S. (2003). *Applied Multiple
    Regression/Correlation Analysis for the Behavioral Sciences* (3rd ed.).
    Erlbaum.
    """
    _check_method(method)
    r1, r2 = np.clip(_as_float(r1), -0.999999, 0.999999), np.clip(_as_float(r2), -0.999999, 0.999999)
    se = np.sqrt(_fisher_se(n1, method) ** 2 + _fisher_se(n2, method) ** 2)
    z = (np.arctanh(r1) - np.arctanh(r2)) / se
    p = 2 * _st.norm.sf(np.abs(z))
    return z, p


def histogram_bins(x: Sequence[float], rule: Literal["fd", "sturges", "scott", "auto"] = "fd") -> np.ndarray:
    """Bin edges from a published bin-width rule.

    * ``"fd"`` – Freedman & Diaconis (1981): width ``2·IQR·n^(-1/3)``; robust
      to outliers.
    * ``"scott"`` – Scott (1979): width ``3.49·σ·n^(-1/3)``.
    * ``"sturges"`` – Sturges (1926): ``log2(n) + 1`` bins; only for small,
      roughly normal samples.
    * ``"auto"`` – NumPy's ``"auto"`` rule, which combines FD and Sturges
      (its exact definition differs between NumPy versions).

    FD breaks down on heavily tied or zero-inflated data: when the IQR is zero
    (or zero up to floating-point noise in the quartiles) its width is zero,
    giving one bin or billions. In that case, and as a safety limit whenever
    FD would need more than 100,000 bins, Sturges is used instead (for
    ``"fd"`` and ``"auto"``). Outliers keep FD unless they are extreme enough
    to push it past that limit.

    On whole-number data (counts, scores, integer dtypes) an FD width below 1
    leaves bins that can hold no value (a comb of empty bars), and a
    fractional width makes bins span different numbers of values (a
    sawtooth). There ``"fd"`` rounds the width to the nearest whole number
    (at least 1) and puts the edges on half-integers, starting at
    ``min(x) - 0.5``, so every value sits inside a bin and every bin spans
    the same number of possible values. The other rules are used as
    published.

    References
    ----------
    Freedman, D., & Diaconis, P. (1981). On the histogram as a density
    estimator: L2 theory. *Z. Wahrscheinlichkeitstheorie verw. Gebiete*, 57,
    453–476.
    Scott, D. W. (1979). On optimal and data-based histograms. *Biometrika*,
    66(3), 605–610.
    """
    return _bin_edges(x, rule)[0]


def _bin_edges(x: Sequence[float], rule: str) -> tuple[np.ndarray, str]:
    """Bin edges and the rule actually used (FD falls back to Sturges when IQR = 0; whole-number widths on integers)."""
    x = _clean(x)
    if x.size == 0:
        raise ValueError("no non-missing values")
    if rule not in ("fd", "sturges", "scott", "auto"):
        raise ValueError(f"rule must be 'fd', 'sturges', 'scott' or 'auto', got {rule!r}")
    if rule in ("fd", "auto") and np.ptp(x) > 0:
        # Decide before calling NumPy, which would try to allocate the huge bin array.
        span = float(np.ptp(x))
        q3, q1 = np.percentile(x, [75, 25])
        iqr = float(q3 - q1)
        if iqr <= 64 * np.finfo(float).eps * max(abs(q1), abs(q3)):  # zero, or ties differing by rounding only
            return np.histogram_bin_edges(x, bins="sturges"), "sturges (IQR is 0, so FD is undefined)"
        width = 2 * iqr * x.size ** (-1 / 3)
        n_bins = span / width
        if n_bins > 100_000:
            return np.histogram_bin_edges(x, bins="sturges"), f"sturges (FD would need {n_bins:,.0f} bins)"
        # Whole numbers (below 2**50, where half-integers are still exact): whole-number widths, half-integer edges.
        if rule == "fd" and np.max(np.abs(x)) < 2**50 and np.all(x == np.round(x)):
            width = max(1.0, float(np.floor(width + 0.5)))
            count = int(np.ceil((span + 1) / width))
            return x.min() - 0.5 + width * np.arange(count + 1), "fd (whole-number widths for integer data)"
    return np.histogram_bin_edges(x, bins=rule), rule


def largest_remainder(values: Sequence[float], total: int) -> np.ndarray:
    """Apportion the integer *total* in proportion to *values*.

    Hamilton's largest-remainder method: take the floor of each exact quota,
    then give the leftover units to the largest fractional remainders. The
    result sums to *total* (unless every value is zero, which returns all
    zeros), unlike independent rounding, which can over- or under-fill a
    waffle chart or a 100 % stacked label set.

    References
    ----------
    Balinski, M. L., & Young, H. P. (1982). *Fair Representation*. Yale
    University Press.
    """
    v = _as_float(values)
    if not np.all(np.isfinite(v)) or np.any(v < 0):
        raise ValueError("values must be finite, non-negative and not missing")
    if v.size:  # rescale by a power of two (exact) so the sum cannot overflow, e.g. for values near 1e308
        v = np.ldexp(v, -int(np.frexp(v.max())[1]))
    if v.sum() == 0:
        return np.zeros(v.size, dtype=int)
    quotas = v / v.sum() * total
    base = np.floor(quotas).astype(int)
    short = int(total - base.sum())
    if short > 0:
        base[np.argsort(-(quotas - base), kind="stable")[:short]] += 1
    return base
