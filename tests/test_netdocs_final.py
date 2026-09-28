"""Tiny and complex network weights, infinite flows, and the documented tie and label rules."""

import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import viz_calc as vc

ROOT = Path(__file__).resolve().parents[1]
TWO_TRIANGLES = [("a", "b", 1), ("b", "c", 1), ("a", "c", 1), ("d", "e", 1), ("e", "f", 1), ("d", "f", 1),
                 ("c", "d", 0.01)]


@pytest.mark.parametrize("scale", [1e-163, 1e-200, 1e-300, 2.0**-1000])
def test_tiny_weights_do_not_underflow_communities_or_layout(scale):
    pytest.importorskip("networkx")
    df = pd.DataFrame(TWO_TRIANGLES, columns=["s", "t", "w"])
    plain = vc.network_map(df, "s", "t", weight="w")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        tiny = vc.network_map(df.assign(w=df["w"] * scale), "s", "t", weight="w")
    assert tiny.table["community"].tolist() == plain.table["community"].tolist() == [0, 0, 0, 1, 1, 1]
    assert tiny.table["betweenness"].tolist() == pytest.approx(plain.table["betweenness"].tolist())
    assert tiny.info["graph"]["a"]["b"]["w"] == scale  # the returned graph keeps the real weights
    assert np.isfinite(np.array(list(tiny.info["positions"].values()))).all()


def test_network_map_rejects_complex_weights():
    pytest.importorskip("networkx")
    df = pd.DataFrame({"s": list("abc"), "t": list("bca"), "w": [1 + 5j, 2, 3]})
    with pytest.raises(TypeError, match="edge weights in 'w' must be real numbers, not complex"):
        vc.network_map(df, "s", "t", weight="w")


@pytest.mark.parametrize("dtype", ["float64", "Float64", "float32"])
def test_sankey_rejects_infinite_flows(dtype):
    pytest.importorskip("plotly")
    df = pd.DataFrame({"s": ["a", "b"], "t": ["b", "c"], "v": pd.array([1.0, np.inf], dtype=dtype)})
    with pytest.raises(ValueError, match="flow values in 'v' must be finite"):
        vc.sankey(df, "s", "t", "v")


def test_sankey_rejects_complex_flows():
    pytest.importorskip("plotly")
    df = pd.DataFrame({"s": ["a", "b"], "t": ["b", "c"], "v": [1.0, 2j]})
    with pytest.raises(TypeError, match="flow values in 'v' must be real numbers, not complex"):
        vc.sankey(df, "s", "t", "v")


def test_sankey_still_accepts_nullable_integer_flows():
    pytest.importorskip("plotly")
    df = pd.DataFrame({"s": ["a", "b"], "t": ["b", "c"], "v": pd.array([1, 2], dtype="Int64")})
    assert vc.sankey(df, "s", "t", "v").table["outflow"].tolist() == [1, 2, 0]


def test_tolerant_slack_is_capped_by_the_last_edge_only():
    # s-y-z-t is 1e-9 longer than s-x-t, a relative 9e-11: within the 1e-10 tolerance. Seen from t, the last
    # edge (y-s, length 1e-9) caps the slack below the difference, so there is no tie; seen from s, the last
    # edge (z-t, length 1) does not, so the paths tie, as the documentation says.
    pytest.importorskip("networkx")
    rows = [("s", "x", 0.1), ("x", "t", 1.0), ("s", "y", 1e9), ("y", "z", 0.1), ("z", "t", 1.0)]
    res = vc.network_map(pd.DataFrame(rows, columns=["a", "b", "w"]), "a", "b", weight="w", communities=False)
    assert res.info["betweenness_arithmetic"] == "tolerant floating-point"
    bc = dict(zip(res.table["node"], res.table["betweenness"]))
    # exact lengths give 1/6 everywhere; the tie, counted from one side only, gives s and y 1/24 more
    assert bc["x"] == pytest.approx(1 / 6)
    assert bc["s"] == pytest.approx(1 / 6 + 1 / 24)
    assert bc["y"] == pytest.approx(1 / 6 + 1 / 24)
    assert all(v == pytest.approx(1 / 6) for v in _exact(rows).values())


def _exact(rows):
    nx = pytest.importorskip("networkx")
    H = nx.Graph()
    H.add_edges_from((u, v, {"d": 1 / Fraction(str(w))}) for u, v, w in rows)
    return nx.betweenness_centrality(H, weight="d")


def test_docs_state_the_last_edge_slack_rule_and_the_small_weight_rescaling():
    methodology = " ".join((ROOT / "docs" / "methodology.md").read_text(encoding="utf-8").split())
    assert "a real extra hop is not a tie" not in methodology
    assert "The cap applies to that edge only" in methodology
    assert "below 2⁻²⁵⁶" in methodology
    doc = " ".join(vc.network_map.__doc__.split())
    assert "real extra hop" not in doc
    assert "The cap applies to the last edge only" in doc
    assert "below 2**-256" in doc
    changelog = " ".join((ROOT / "CHANGELOG.md").read_text(encoding="utf-8").split())
    assert "never merging a real extra hop" not in changelog


def test_conventions_document_the_zero_false_label_limitation():
    text = " ".join((ROOT / "docs" / "getting-started.md").read_text(encoding="utf-8").split())
    assert "treats the labels 0 and False, and 1 and True, as the same key" in text
    d = pd.DataFrame(np.random.default_rng(1).normal(size=(20, 2)), columns=[0, False])
    assert d[0].shape == (20, 2)  # pandas itself returns both columns for either label
    with pytest.raises(ValueError, match="different columns"):
        vc.quadrant_plot(d, x=0, y=False)
