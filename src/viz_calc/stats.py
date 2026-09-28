"""Statistical building blocks used by the plots.

Each function is a small, tested implementation of a published method, so the
numbers a chart shows can be traced back to a citation. They are public: use
them on their own when you want the numbers without the picture.

References are listed in each docstring and collected on the *Methodology*
page of the documentation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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


def _clean(x: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def wilson_ci(successes: int | np.ndarray, n: int | np.ndarray, level: float = 0.95) -> tuple[Any, Any]:
    """Wilson score interval for a binomial proportion.

    Preferred over the textbook Wald interval, which has poor coverage for
    small *n* or proportions near 0 or 1 (Brown, Cai & DasGupta, 2001).

    Returns ``(lower, upper)`` as proportions: floats for scalar input,
    arrays for array input. ``n == 0`` gives ``nan``.

    References
    ----------
    Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. *JASA*, 22(158), 209–212.
    Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a
    binomial proportion. *Statistical Science*, 16(2), 101–133.
    """
    k = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    z = _st.norm.ppf(1 - (1 - level) / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lower = np.where(n > 0, np.clip(centre - half, 0, 1), np.nan)
    upper = np.where(n > 0, np.clip(centre + half, 0, 1), np.nan)
    if lower.ndim == 0:
        return float(lower), float(upper)
    return lower, upper


def mean_ci(x: Sequence[float], level: float = 0.95) -> tuple[float, float, float]:
    """Mean and its t-based confidence interval: ``(mean, lower, upper)``.

    Uses the Student *t* distribution with ``n - 1`` degrees of freedom. With
    fewer than two observations the interval is ``nan``.
    """
    x = _clean(x)
    n = x.size
    if n == 0:
        return (np.nan, np.nan, np.nan)
    m = float(x.mean())
    if n < 2:
        return (m, np.nan, np.nan)
    se = x.std(ddof=1) / np.sqrt(n)
    t = _st.t.ppf(1 - (1 - level) / 2, n - 1)
    return (m, m - t * se, m + t * se)


def welch_test(a: Sequence[float], b: Sequence[float], level: float = 0.95) -> dict[str, float]:
    """Welch's unequal-variance *t*-test for ``mean(b) - mean(a)``.

    Welch's test keeps its error rate when group variances or sizes differ and
    loses almost nothing when they are equal, so it is the recommended default
    over Student's *t*-test (Delacre, Lakens & Leys, 2017).

    Returns a dict with ``difference``, ``ci_low``, ``ci_high``, ``t``,
    ``df`` (Welch–Satterthwaite) and ``p``.

    References
    ----------
    Welch, B. L. (1947). The generalization of 'Student's' problem when several
    different population variances are involved. *Biometrika*, 34, 28–35.
    Delacre, M., Lakens, D., & Leys, C. (2017). Why psychologists should by
    default use Welch's t-test instead of Student's t-test. *International
    Review of Social Psychology*, 30(1), 92–101.
    """
    a, b = _clean(a), _clean(b)
    if a.size < 2 or b.size < 2:
        raise ValueError("each group needs at least two non-missing values")
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    diff = b.mean() - a.mean()
    se = np.sqrt(va + vb)
    if se == 0:
        return {"difference": float(diff), "ci_low": float(diff), "ci_high": float(diff),
                "t": np.inf if diff else np.nan, "df": float(a.size + b.size - 2), "p": 0.0 if diff else 1.0}
    df = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    t = diff / se
    p = 2 * _st.t.sf(abs(t), df)
    crit = _st.t.ppf(1 - (1 - level) / 2, df)
    return {"difference": float(diff), "ci_low": float(diff - crit * se), "ci_high": float(diff + crit * se),
            "t": float(t), "df": float(df), "p": float(p)}


def hedges_g(a: Sequence[float], b: Sequence[float], level: float = 0.95) -> dict[str, float]:
    """Hedges' *g* standardized mean difference for ``b - a`` with a CI.

    Cohen's *d* with the pooled SD, multiplied by the small-sample correction
    ``J = 1 - 3 / (4 * df - 1)``. The interval uses the large-sample variance
    from Hedges & Olkin (1985).

    References
    ----------
    Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect
    size and related estimators. *Journal of Educational Statistics*, 6(2),
    107–128.
    Hedges, L. V., & Olkin, I. (1985). *Statistical Methods for
    Meta-Analysis*. Academic Press.
    """
    a, b = _clean(a), _clean(b)
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        raise ValueError("each group needs at least two non-missing values")
    df = n1 + n2 - 2
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / df)
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
    (Ho et al., 2019). A fixed *seed* keeps figures reproducible.

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
    p = np.asarray(p, dtype=float)
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
    elif method == "fdr_bh":
        stepped = np.minimum.accumulate((m / np.arange(m, 0, -1)) * ranked[::-1])[::-1]
        adj = np.empty(m)
        adj[order] = np.minimum(stepped, 1.0)
    else:
        raise ValueError("method must be 'holm', 'fdr_bh', 'bonferroni' or 'none'")
    out[mask] = adj
    return out


def _fisher_se(n: np.ndarray | float, method: str) -> np.ndarray | float:
    # Spearman's rho has a larger sampling variance than Pearson's r; the
    # 1.06 / (n - 3) approximation is from Fieller, Hartley & Pearson (1957).
    # The standard error is undefined for n <= 3; return nan rather than inf/0.
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 3, np.sqrt((1.06 if method == "spearman" else 1.0) / (n - 3)), np.nan)


def _check_method(method: str) -> None:
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")


def correlation_test(x: Sequence[float], y: Sequence[float], method: Literal["pearson", "spearman"] = "pearson",
                     level: float = 0.95) -> dict[str, float]:
    """Correlation with p-value and Fisher-*z* confidence interval.

    Pairs with a missing value in either variable are dropped. Returns
    ``r``, ``ci_low``, ``ci_high``, ``p`` and ``n``.

    References
    ----------
    Fisher, R. A. (1921). On the "probable error" of a coefficient of
    correlation deduced from a small sample. *Metron*, 1, 3–32.
    Fieller, E. C., Hartley, H. O., & Pearson, E. S. (1957). Tests for rank
    correlation coefficients. I. *Biometrika*, 44(3/4), 470–481.
    """
    _check_method(method)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    n = x.size
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"r": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": n}
    res = _st.spearmanr(x, y) if method == "spearman" else _st.pearsonr(x, y)
    r, p = float(res[0]), float(res[1])
    if n > 3 and abs(r) < 1:
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
    result is ``nan`` wherever either ``n`` is 3 or less.

    References
    ----------
    Fisher, R. A. (1921). *Metron*, 1, 3–32.
    Cohen, J., Cohen, P., West, S. G., & Aiken, L. S. (2003). *Applied Multiple
    Regression/Correlation Analysis for the Behavioral Sciences* (3rd ed.).
    Erlbaum.
    """
    _check_method(method)
    r1, r2 = np.clip(np.asarray(r1, dtype=float), -0.999999, 0.999999), np.clip(np.asarray(r2, dtype=float), -0.999999, 0.999999)
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
    (or tiny relative to the range) its width is zero or near zero, giving
    one bin or millions of bins. Whenever FD would give more bins than there
    are observations, Sturges is used instead (for ``"fd"`` and ``"auto"``).

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
    """Bin edges and the rule actually used (FD falls back to Sturges when IQR = 0)."""
    x = _clean(x)
    if x.size == 0:
        raise ValueError("no non-missing values")
    if rule not in ("fd", "sturges", "scott", "auto"):
        raise ValueError(f"rule must be 'fd', 'sturges', 'scott' or 'auto', got {rule!r}")
    if rule in ("fd", "auto") and np.ptp(x) > 0:
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        width = 2 * iqr * x.size ** (-1 / 3)
        # Decide before calling NumPy, which would try to allocate the huge bin array.
        if width <= 0 or np.ptp(x) / width > x.size:
            reason = "IQR = 0" if iqr == 0 else "IQR near 0"
            return np.histogram_bin_edges(x, bins="sturges"), f"sturges ({reason}, FD degenerate)"
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
    v = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(v)) or np.any(v < 0):
        raise ValueError("values must be finite, non-negative and not missing")
    if v.sum() == 0:
        return np.zeros(v.size, dtype=int)
    quotas = v / v.sum() * total
    base = np.floor(quotas).astype(int)
    short = int(total - base.sum())
    if short > 0:
        base[np.argsort(-(quotas - base), kind="stable")[:short]] += 1
    return base
