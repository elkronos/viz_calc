"""Regression tests for the thirteenth review round: weighted betweenness."""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import viz_calc as vc

nx = pytest.importorskip("networkx")


def _bc(frame, **kw):
    import matplotlib.pyplot as plt

    res = vc.network_map(frame, "s", "t", weight="w", communities=False, **kw)
    plt.close(res.figure)
    return res, res.table.set_index("node")["betweenness"]


def _truth(rows, directed=False):
    H = nx.DiGraph() if directed else nx.Graph()
    H.add_nodes_from({u for u, _, _ in rows} | {v for _, v, _ in rows})
    H.add_weighted_edges_from(((u, v, 1 / Fraction(w)) for u, v, w in rows if u != v), weight="d")
    return {k: float(v) for k, v in nx.betweenness_centrality(H, weight="d").items()}


@pytest.mark.parametrize("k,dtype", [(5_600_003, "float32"), (703, "float16")])
def test_counts_rounded_by_a_low_precision_type_keep_their_ties(k, dtype):
    # 1/(3k) + 1/(6k) == 1/(2k), but float32/float16 cannot hold 3k and 6k exactly
    e = pd.DataFrame([("A", "B", 3 * k), ("B", "C", 6 * k), ("A", "C", 2 * k)], columns=["s", "t", "w"])
    res, bc = _bc(e.astype({"w": dtype}))
    assert res.info["betweenness_arithmetic"] == "tolerant floating-point"
    assert bc["B"] == pytest.approx(0.5)


@pytest.mark.parametrize("value,dtype,exact", [
    (1000000.25, "float32", False),  # 9 digits: may be the rounding of 1000000.26
    (16777215.0, "float32", True), (16777216.0, "float32", False),
    (0.125, "float16", True), (1.375, "float16", False),
    (1001.5, "float64", True), (0.3, "float64", False), (2.0**53, "float64", False),
])
def test_which_floats_count_as_exact(value, dtype, exact):
    from viz_calc.network import _exact_weights

    assert (_exact_weights(np.array([value], dtype=dtype)) is not None) is exact


def test_exact_paths_that_differ_by_less_than_float_rounding():
    c = 10**12  # A-B-C is longer than A-C by 5e-13 of its length: below float64 resolution, but not a tie
    near = [("A", "B", 2 * c - 1), ("B", "C", 2 * c - 1), ("A", "C", c)]
    res, bc = _bc(pd.DataFrame(near, columns=["s", "t", "w"]))
    assert res.info["betweenness_arithmetic"] == "exact" and bc["B"] == 0
    k = 10**15  # a true tie between huge counts
    tie = [("A", "B", 3 * k), ("B", "C", 6 * k), ("A", "C", 2 * k)]
    assert _bc(pd.DataFrame(tie, columns=["s", "t", "w"]))[1]["B"] == pytest.approx(0.5)


@pytest.mark.parametrize("directed", [False, True])
def test_exact_betweenness_matches_rational_networkx(directed):
    rng = np.random.default_rng(11)
    for trial in range(12):
        if trial % 3 == 0:  # grid: many equally short paths
            rows = [((i, j), (i + di, j + dj)) for i in range(5) for j in range(5) for di, dj in ((0, 1), (1, 0))
                    if i + di < 5 and j + dj < 5]
        elif trial % 3 == 1:  # long thin tree plus a few shortcuts
            rows = [(int(rng.integers(max(0, i - 3), i)), i) for i in range(1, 40)] + [(0, 20), (5, 39)]
        else:
            rows = [(i, j) for i in range(14) for j in range(14) if (i != j if directed else i < j) and rng.random() < 0.2]
        ws = rng.choice([1, 2, 3, 4, 6, 12], len(rows)) if trial % 2 else rng.integers(1, 10**6, len(rows))
        rows = [(str(u), str(v), int(w)) for (u, v), w in zip(rows, ws)]
        e = pd.DataFrame(rows, columns=["s", "t", "w"])
        truth = _truth(rows, directed)
        res, bc = _bc(e, directed=directed)
        assert res.info["betweenness_arithmetic"] == "exact"
        assert all(bc[n] == pytest.approx(truth[n], abs=1e-12) for n in truth)
        shuffled = _bc(e.sample(frac=1, random_state=trial), directed=directed)[1]
        pd.testing.assert_series_equal(bc.sort_index(), shuffled.sort_index(), check_exact=False, atol=1e-12, rtol=0)


def test_dyadic_float_weights_are_exact():
    rows = [("A", "B", 0.5), ("B", "D", 0.25), ("A", "C", 0.25), ("C", "D", 0.5), ("D", "E", 1.5)]
    res, bc = _bc(pd.DataFrame(rows, columns=["s", "t", "w"]))
    assert res.info["betweenness_arithmetic"] == "exact"
    assert all(bc[n] == pytest.approx(v, abs=1e-12) for n, v in _truth(rows).items())


def test_weights_too_small_or_summing_too_large_raise():
    tiny = pd.DataFrame({"s": ["A", "B"], "t": ["B", "C"], "w": [5e-324, 1.0]})
    with pytest.raises(ValueError, match="too small"):
        vc.network_map(tiny, "s", "t", weight="w")
    huge = pd.DataFrame({"s": ["A", "A"], "t": ["B", "B"], "w": [1.7e308, 1.7e308]})
    with pytest.raises(ValueError, match="largest float"):
        vc.network_map(huge, "s", "t", weight="w")


def test_edges_too_short_for_float64_to_resolve():
    # 1/2**60 is below float64 resolution next to a path of length 1, so floats cannot order B, C and D
    rows = [("A", "B", 1), ("B", "C", 2**60), ("B", "D", 2**61), ("D", "C", 2**61)]
    res, bc = _bc(pd.DataFrame(rows, columns=["s", "t", "w"]))
    assert res.info["betweenness_arithmetic"] == "exact"
    assert all(bc[n] == pytest.approx(v, abs=1e-12) for n, v in _truth(rows).items())
