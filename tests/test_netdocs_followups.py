"""Networks, flows and the documentation's claims about them."""

import re
import warnings
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import viz_calc as vc

ROOT = Path(__file__).resolve().parents[1]


def _truth(rows, directed=False):
    nx = pytest.importorskip("networkx")
    H = nx.DiGraph() if directed else nx.Graph()
    H.add_nodes_from({u for u, _, _ in rows} | {v for _, v, _ in rows})
    H.add_weighted_edges_from(((u, v, 1 / Fraction(w)) for u, v, w in rows if u != v), weight="d")
    return {k: float(v) for k, v in nx.betweenness_centrality(H, weight="d").items()}


@pytest.mark.parametrize("column", ["s", "t"])
def test_sankey_rejects_rows_with_a_missing_endpoint(column):
    pytest.importorskip("plotly")
    df = pd.DataFrame({"s": ["a", "a", "a"], "t": ["b", "c", "c"], "v": [1, 2, 3]})
    df.loc[1, column] = None
    with pytest.raises(ValueError, match=f"1 row\\(s\\) have no '{column}' value"):
        vc.sankey(df, "s", "t", "v")


def test_sankey_categorical_endpoints_with_unused_levels_are_fine():
    pytest.importorskip("plotly")
    df = pd.DataFrame({"s": pd.Categorical(["a", "b"], categories=["a", "b", "z"]), "t": ["b", "c"], "v": [1, 2]})
    assert vc.sankey(df, "s", "t", "v").table["inflow"].tolist() == [0, 1, 2]


@pytest.mark.parametrize("kwargs,match", [
    ({"source": "s", "target": "s", "weight": "w"}, "source and target"),
    ({"source": "s", "target": "t", "weight": "s"}, "source and weight"),
    ({"source": "s", "target": "t", "weight": "t"}, "target and weight"),
])
def test_network_map_rejects_one_column_for_two_roles(kwargs, match):
    pytest.importorskip("networkx")
    df = pd.DataFrame({"s": [1, 2, 3], "t": [2, 3, 1], "w": [1, 2, 3]})
    with pytest.raises(ValueError, match=match):
        vc.network_map(df, **kwargs)


def test_negative_weight_on_a_row_left_out_is_ignored():
    pytest.importorskip("networkx")
    df = pd.DataFrame({"s": ["a", "b", None], "t": ["b", "c", "c"], "w": [1.0, 2.0, -1.0]})
    res = vc.network_map(df, "s", "t", weight="w")
    assert res.info["graph"].number_of_edges() == 2
    with pytest.raises(ValueError, match="non-negative"):
        vc.network_map(df.fillna({"s": "a"}), "s", "t", weight="w")


def test_path_lengths_past_the_largest_float_raise():
    pytest.importorskip("networkx")
    # a-b-c is 2e308 long and a-x-c 1.8e308: each edge fits in a float, neither path does
    w = [1e-308, 1e-308, 1 / 0.9e308, 1 / 0.9e308]
    df = pd.DataFrame({"s": ["a", "b", "a", "x"], "t": ["b", "c", "x", "c"], "w": w})
    with pytest.raises(ValueError, match="longer than the largest float"):
        vc.network_map(df, "s", "t", weight="w", communities=False)
    # the same graph rescaled has the right betweenness
    res = vc.network_map(df.assign(w=df["w"] * 1e10), "s", "t", weight="w", communities=False)
    bc = res.table.set_index("node")["betweenness"]
    assert bc["b"] == 0 and bc["x"] == pytest.approx(1 / 3)


def test_one_very_heavy_edge_is_resolved_locally(monkeypatch):
    pytest.importorskip("networkx")
    import viz_calc.network as network

    sizes = []
    original = network._rational_shortest_paths
    monkeypatch.setattr(network, "_rational_shortest_paths",
                        lambda start, into: (sizes.append(len(into)), original(start, into))[1])
    rng = np.random.default_rng(0)
    n = 40
    rows = [(str(i), str(int(rng.integers(0, i))), int(rng.integers(1, 1000))) for i in range(1, n)]
    rows += [(str(int(a)), str(int(b)), int(rng.integers(1, 1000))) for a, b in rng.integers(0, n, (40, 2)) if a != b]
    rows.append(("0", "1", 10**18))  # too short (1e-18) for float64 to separate its ends from any source
    res = vc.network_map(pd.DataFrame(rows, columns=["s", "t", "w"]), "s", "t", weight="w", communities=False)
    assert res.info["betweenness_arithmetic"] == "exact"
    assert sizes and max(sizes) <= 2  # only the two float-tied ends go through rational arithmetic
    bc = res.table.set_index("node")["betweenness"]
    assert all(bc[k] == pytest.approx(v, abs=1e-12) for k, v in _truth(rows).items())


@pytest.mark.parametrize("directed", [False, True])
def test_tiny_edges_match_rational_networkx(directed):
    pytest.importorskip("networkx")
    rng = np.random.default_rng(4)
    for _ in range(30):
        n = int(rng.integers(5, 14))
        pairs = [(i, j) for i in range(n) for j in range(n)
                 if (i != j if directed else i < j) and rng.random() < 0.3]
        rows = [(str(u), str(v), int(rng.integers(1, 5)) * int(rng.choice([1, 1, 2**55, 2**60])))
                for u, v in pairs]
        if not rows:
            continue
        res = vc.network_map(pd.DataFrame(rows, columns=["s", "t", "w"]), "s", "t", weight="w",
                             directed=directed, communities=False)
        plt.close(res.figure)
        bc = res.table.set_index("node")["betweenness"]
        assert all(bc[k] == pytest.approx(v, abs=1e-12) for k, v in _truth(rows, directed).items())


def test_unordered_groups_are_sorted_as_the_conventions_say():
    df = pd.DataFrame({"arm": ["placebo"] * 3 + ["drug"] * 3 + ["alt"] * 3, "y": np.arange(9.0)})
    assert vc.estimation_plot(df, "arm", "y", n_resamples=100).info["reference"] == "alt"
    text = (ROOT / "docs/getting-started.md").read_text(encoding="utf-8")
    order = text.split("* **Order.**")[1].split("\n* ")[0]
    assert "first appearance" not in order and "sorted" in order and "reference" in order


def test_conventions_name_waterfall_start_label():
    conventions = (ROOT / "docs/getting-started.md").read_text(encoding="utf-8").split("## Conventions")[1]
    assert re.search(r"`waterfall`,\s+`start_label`", conventions)


def test_gallery_claims_match_its_contents():
    gallery = ROOT / "docs/gallery.md"
    if not gallery.exists():
        pytest.skip("gallery page not generated")
    shown = set(re.findall(r"### `(\w+)`", gallery.read_text(encoding="utf-8")))
    charts = [n for n in vc.__all__ if callable(getattr(vc, n)) and n not in {"VizResult", "palette", "pca", "to_pptx"}]
    missing = [n for n in charts if n not in shown]
    for page in ("README.md", "docs/index.md"):
        text = (ROOT / page).read_text(encoding="utf-8")
        assert "every chart" not in text
        line = next(par for par in text.split("\n\n") if "allery" in par and "every" not in par and "`" in par)
        assert all(f"`{n}`" in line for n in missing), (page, missing)


@pytest.mark.parametrize("rows,expected", [
    # the middle edge (1e-17) is too short for float64 to add to a path of length 10
    ([("s", "a", 0.1), ("a", "b", 1e17), ("b", "c", 0.1)], {"s": 0, "a": 2 / 3, "b": 2 / 3, "c": 0}),
    # a second, even shorter edge inside the same float-tied run
    ([("s", "a", 0.1), ("a", "b", 1e17), ("b", "x", 1e300), ("x", "c", 0.1)],
     {"s": 0, "a": 0.5, "b": 2 / 3, "x": 0.5, "c": 0}),
])
def test_tolerant_paths_through_a_too_short_edge_still_count(rows, expected):
    pytest.importorskip("networkx")
    df = pd.DataFrame(rows, columns=["s", "t", "w"])
    res = vc.network_map(df, "s", "t", weight="w", communities=False)
    assert res.info["betweenness_arithmetic"] == "tolerant floating-point"
    bc = res.table.set_index("node")["betweenness"]
    truth = _truth([(u, v, Fraction(repr(w))) for u, v, w in rows])
    assert truth == pytest.approx(expected)
    assert all(bc[k] == pytest.approx(v, abs=1e-12) for k, v in truth.items())


@pytest.mark.parametrize("directed", [False, True])
def test_tolerant_float_tied_runs_keep_every_path_on_trees(directed):
    pytest.importorskip("networkx")
    rng = np.random.default_rng(7)
    for _ in range(30):
        n = int(rng.integers(4, 12))
        # a tree has one path per pair, so the answer only depends on no path being lost
        edges = [(str(i), str(int(rng.integers(0, i)))) for i in range(1, n)]
        if directed:
            edges = [(v, u) if rng.random() < 0.5 else (u, v) for u, v in edges]
        rows = [(u, v, float(rng.choice([0.1, 0.3, 0.7, 1.1]) * rng.choice([1, 1e17, 1e30, 1e300])))
                for u, v in edges]
        res = vc.network_map(pd.DataFrame(rows, columns=["s", "t", "w"]), "s", "t", weight="w",
                             directed=directed, communities=False)
        plt.close(res.figure)
        bc = res.table.set_index("node")["betweenness"]
        truth = _truth([(u, v, 1) for u, v, _ in rows], directed)
        assert all(bc[k] == pytest.approx(v, abs=1e-12) for k, v in truth.items())


def test_very_large_weights_do_not_overflow_communities_or_layout():
    pytest.importorskip("networkx")
    rows = [("a", "b", 2), ("b", "c", 1), ("c", "a", 3), ("d", "e", 2), ("e", "f", 1), ("f", "d", 3), ("c", "d", 0.01)]
    df = pd.DataFrame(rows, columns=["s", "t", "w"])
    small = vc.network_map(df, "s", "t", weight="w")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        big = vc.network_map(df.assign(w=df["w"] * 2.0**600), "s", "t", weight="w")
    assert big.table["community"].tolist() == small.table["community"].tolist()
    assert big.info["n_communities"] == 2
    assert big.info["graph"]["a"]["b"]["w"] == 2 * 2.0**600  # the returned graph keeps the real weights
    assert np.isfinite(np.array(list(big.info["positions"].values()))).all()


def test_node_strength_past_the_largest_float_raises():
    pytest.importorskip("networkx")
    df = pd.DataFrame({"s": list("abcd"), "t": list("bcda"), "w": [1e308] * 4})
    with pytest.raises(ValueError, match="weights at node 'a' add up to more than the largest float"):
        vc.network_map(df, "s", "t", weight="w", communities=False)


def test_dataset_docstrings_name_every_column():
    from viz_calc import datasets

    for make in (datasets.trial, datasets.sales, datasets.measurements, datasets.projects, datasets.network):
        for column in make().columns:
            assert f"``{column}``" in make.__doc__, (make.__name__, column)
    survey = datasets.survey()
    assert all(f"``{c}``" in datasets.survey.__doc__ for c in ("team", "tenure_years", "remote"))
    assert survey.shape[1] == 3 + 5


def test_conventions_name_the_order_exceptions():
    text = (ROOT / "docs/getting-started.md").read_text(encoding="utf-8")
    order = text.split("* **Order.**")[1].split("\n* ")[0]
    for name in ("waffle", "divergent_bar", "nested_pie", "funnel", "waterfall", "bullet", "lollipop", "dumbbell",
                 "likert", "circular_bar", "gantt", "duration_plot", "benchmark_bar"):
        assert f"`{name}`" in order, name
    frame = pd.DataFrame({"c": ["zeta", "alpha", "mid"], "l": [1, 2, 3], "r": [3, 2, 1]})
    assert vc.divergent_bar(frame, category="c", left="l", right="r").table["c"].tolist() == ["zeta", "alpha", "mid"]
    nested = vc.nested_pie(pd.DataFrame({"o": ["z", "a"], "i": ["y", "b"]}), outer="o", inner="i")
    assert nested.table["o"].tolist() == ["z", "a"]
