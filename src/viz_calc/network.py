"""Networks and flows (optional dependencies: ``networkx``, ``plotly``)."""

from __future__ import annotations

import decimal
import heapq
import math
from fractions import Fraction
from itertools import count
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from ._core import NEUTRAL, VizResult, check_dataframe, check_numeric, palette, require

__all__ = ["network_map", "sankey"]


EXACT_EDGE_LIMIT = 2000  # above this, exact rational betweenness gets slow; the tolerant float version is used


def _exact_weights(values: np.ndarray) -> list[Fraction] | None:
    """Exact values for weights that are exactly what they display, else ``None``.

    Integers, booleans, ``Decimal`` and ``Fraction`` objects qualify, and so do
    floats whose stored binary value equals the shortest decimal that prints
    it (``2.0``, ``0.125``, ``1001.5``). Any other float (``0.3``, ``73/9``,
    ``count/438``) stands for a value its type cannot hold, and which one was
    meant is ambiguous, so such columns use the tolerant comparison instead.
    """
    out = []
    for v in values:
        if isinstance(v, (bool, np.bool_, int, np.integer)):
            out.append(Fraction(int(v)))
        elif isinstance(v, (Fraction, decimal.Decimal)):
            out.append(Fraction(v))
        elif isinstance(v, (float, np.floating)):
            stored = Fraction(*v.as_integer_ratio())
            if stored != Fraction(np.format_float_positional(v, unique=True, trim="-")):
                return None
            out.append(stored)
        else:
            return None
    return out


def _rounding_tolerance(values: np.ndarray) -> float:
    """Relative rounding error of path lengths built from these weights: 4 machine epsilons of their float type.

    float32 and float16 hold only ~7 and ~3 significant digits, so a value
    entered and then rescaled can be off by a whole epsilon; four absorb it.
    The floor of 64 float64 epsilons covers the additions along a path.
    """
    if values.dtype == object:
        eps = [np.finfo(t).eps for t in {type(v) for v in values if isinstance(v, np.floating)}]
    else:
        eps = [np.finfo(values.dtype).eps] if values.dtype.kind == "f" else []
    return max([64 * float(np.finfo(np.float64).eps)] + [4 * float(e) for e in eps])


def _weighted_betweenness(G, distance: dict, rel_tol: float = 1e-10, rounding: float = 0.0) -> dict:
    """Brandes' betweenness centrality with ties anchored on the true shortest distance.

    ``distance[(u, v)]`` is the length of edge u→v (for undirected graphs, both
    orientations are present); edges missing from it are not traversed. For
    each source, Dijkstra first finds the shortest distances, which do not
    depend on edge order. Then v is a predecessor of w on a shortest path when
    ``dist[v] < dist[w]`` and ``dist[v] + len(v, w)`` exceeds ``dist[w]`` by at
    most ``max(rounding, min(rel_tol, len(v, w) / (4 * dist[w]))) * dist[w]``.
    Paths equal up to a relative *rel_tol* (1e-10, as in igraph) count as
    equally short (1/2 + 1/12 vs 1/3 + 1/4, 0.3, 73/9, count/total shares),
    but that slack never exceeds a quarter of the edge being tested, so a real
    extra hop is not a tie. Only differences within the weights' own
    *rounding* error, which low-precision floats cannot resolve, always tie.
    The result does not depend on row order or on edges elsewhere.
    Normalized as in NetworkX: by 1/((n-1)(n-2)).

    References
    ----------
    Brandes, U. (2001). A faster algorithm for betweenness centrality.
    *Journal of Mathematical Sociology*, 25(2), 163–177.
    """
    nodes = list(G)
    out = {v: [(w, distance[(v, w)]) for w in G[v] if (v, w) in distance] for v in nodes}
    bc = dict.fromkeys(nodes, 0.0)
    for s in nodes:
        dist, done, heap, counter = {s: 0.0}, set(), [(0.0, 0, s)], count(1)
        while heap:  # 1. shortest distances
            d, _, v = heapq.heappop(heap)
            if v in done:
                continue
            done.add(v)
            for w, length in out[v]:
                if w not in dist or d + length < dist[w]:
                    dist[w] = d + length
                    heapq.heappush(heap, (dist[w], next(counter), w))
        preds = {v: [] for v in dist}  # 2. predecessors on any path within tolerance of the shortest
        for v in dist:
            for w, length in out[v]:
                # beyond the data's own rounding, the slack never exceeds a quarter of the edge being tested
                slack = max(rounding * dist[w], min(rel_tol * dist[w], 0.25 * length))
                if dist[v] < dist[w] and dist[v] + length <= dist[w] + slack:
                    preds[w].append(v)
        order = sorted(dist, key=dist.get)  # predecessors are always strictly closer, so this is topological
        sigma = dict.fromkeys(order, 0.0)
        sigma[s] = 1.0
        for w in order:
            if w != s:
                sigma[w] = sum(sigma[v] for v in preds[w])
        delta = dict.fromkeys(order, 0.0)  # 3. Brandes accumulation
        for w in reversed(order):
            if sigma[w] > 0:
                for v in preds[w]:
                    delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    n = len(nodes)
    scale = 1 / ((n - 1) * (n - 2)) if n > 2 else 0.0
    return {v: b * scale for v, b in bc.items()}


def network_map(
    data: pd.DataFrame,
    source: str,
    target: str,
    weight: str | None = None,
    directed: bool = False,
    communities: bool = True,
    size_by: Literal["degree", "strength", "betweenness"] = "degree",
    labels: bool = True,
    interactive: bool = False,
    seed: int = 0,
    ax: Axes | None = None,
) -> VizResult:
    """Network graph from an edge list, with centrality and community detection.

    Node metrics are computed from the graph itself: degree, strength
    (weighted degree), betweenness centrality and, if *communities* is on,
    Louvain communities (Blondel et al., 2008), using NetworkX's built-in
    implementation. Node size encodes *size_by*; colour encodes community.
    The layout is seeded so it is reproducible. Repeated edges (including
    B→A after A→B in an undirected graph) are merged, summing their weights;
    missing weights raise an error. Weighted betweenness uses 1/weight as edge
    length. Weights that are exactly what they display (integers, or floats
    such as 0.5 or 1001.25) are compared in exact rational arithmetic. Other
    floats (0.3, 73/9, shares such as count/438) cannot be stored exactly, so
    path lengths within a relative 1e-10 of each other count as equal, as in
    igraph, unless the difference is more than a quarter of a path's last
    edge (a real extra hop). float32 and float16 columns hold only ~7 and ~3
    digits, so there any difference within four machine epsilons (4.8e-7 and
    0.4%) counts as equal. Either way equally short paths share credit and
    the result does not depend on row order.
    ``info["betweenness_arithmetic"]`` says which was used, and
    ``info["betweenness_tolerance"]`` gives the tolerance (graphs over 2,000
    edges always use the tolerant version, for speed). For directed graphs,
    communities are found on the undirected graph with reciprocal weights
    summed.

    Set ``interactive=True`` for a Plotly figure with hover details.

    References
    ----------
    Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008).
    Fast unfolding of communities in large networks. *J. Stat. Mech.*, P10008.
    Brandes, U. (2001). A faster algorithm for betweenness centrality.
    *Journal of Mathematical Sociology*, 25(2), 163–177.
    """
    nx = require("networkx", "network")
    check_dataframe(data, [source, target, weight])
    weighted = weight is not None  # a column may legitimately be labelled 0
    if weighted:
        check_numeric(data, weight)
        if (data[weight] < 0).any():
            raise ValueError("edge weights must be non-negative")
    # Work on plain Python objects: categorical columns, non-string labels and nullable dtypes all behave the same.
    src = data[source].to_numpy(dtype=object)
    tgt = data[target].to_numpy(dtype=object)
    keep = ~(pd.isna(src) | pd.isna(tgt))
    wts = data[weight].to_numpy(dtype=float, na_value=np.nan) if weighted else np.ones(len(data))
    if weighted and np.isnan(wts[keep]).any():
        raise ValueError(f"edge weights in {weight!r} contain missing values; drop or fill them first")
    if weighted and np.isinf(wts[keep]).any():
        raise ValueError(f"edge weights in {weight!r} must be finite")
    # Repeated edges are combined (weights summed) instead of letting the last row win; each edge keeps the
    # orientation it was first seen with. Sums are exact when the weights are (see _exact_weights), otherwise
    # correctly rounded (math.fsum, so 9 x 73/9 == 73).
    numpy_dtype = getattr(data[weight].dtype, "numpy_dtype", None) if weighted else None
    if numpy_dtype is not None and pd.api.types.is_float_dtype(numpy_dtype):  # nullable Float32 -> float32 scalars
        raw = data[weight].to_numpy(dtype=numpy_dtype, na_value=np.nan)
    else:
        raw = data[weight].to_numpy() if weighted else np.ones(len(data), dtype=np.int64)
    exact_values = _exact_weights(raw[keep]) if weighted else None
    values = exact_values if exact_values is not None else [float(x) for x in wts[keep]]
    merged: dict = {}
    for u, v, w in zip(src[keep], tgt[keep], values):
        key = (u, v) if directed else frozenset((u, v))
        if key in merged:
            merged[key][2].append(w)
        else:
            merged[key] = [u, v, [w]]
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(pd.unique(np.column_stack([src[keep], tgt[keep]]).ravel()))  # first-appearance order
    edge_weight = {}
    for u, v, ws in merged.values():
        G.add_edge(u, v)
        if weighted:
            total = sum(ws, Fraction(0)) if exact_values is not None else math.fsum(ws)
            G[u][v][weight] = float(total)
            edge_weight[(u, v)] = total
    wkey = weight if weighted else None
    nodes = list(G.nodes())
    table = pd.DataFrame({"node": nodes})
    table["degree"] = [G.degree(n) for n in nodes]
    table["strength"] = [G.degree(n, weight=wkey) for n in nodes]
    # Betweenness treats weights as distances; stronger ties should be shorter, so use 1/weight; zero-weight
    # edges carry no tie and are left out. Weights stored exactly are compared exactly (Fraction distances in
    # NetworkX), so equally short paths always tie; other floats, whose intended exact value cannot be known, use
    # Brandes with a tolerance that absorbs rounding but never a real extra hop.
    method = tolerance = None
    if weighted:
        positive = {(u, v): w for (u, v), w in edge_weight.items() if w > 0 and u != v}
        if exact_values is not None and G.number_of_edges() <= EXACT_EDGE_LIMIT:
            method = "exact"
            H = G.__class__()
            H.add_nodes_from(G)
            H.add_edges_from((u, v, {"distance": 1 / w}) for (u, v), w in positive.items())
            bc = {n: float(b) for n, b in nx.betweenness_centrality(H, weight="distance").items()}
        else:
            method = "tolerant floating-point"
            rounding = _rounding_tolerance(raw[keep] if exact_values is None else np.array([]))
            tolerance = max(1e-10, rounding)
            distance = {}
            for (u, v), w in positive.items():
                distance[(u, v)] = 1.0 / float(w)
                if not directed:
                    distance[(v, u)] = 1.0 / float(w)
            bc = _weighted_betweenness(G, distance, rounding=rounding)
    else:
        bc = nx.betweenness_centrality(G)
    table["betweenness"] = [bc[n] for n in nodes]
    if communities:
        # Communities are found on the undirected graph; reciprocal edges A→B and B→A add their weights.
        U = nx.Graph()
        U.add_nodes_from(G)
        for u, v, d in G.edges(data=True):
            w = d[weight] if weighted else 1.0
            if U.has_edge(u, v):
                U[u][v]["w"] += w
            else:
                U.add_edge(u, v, w=w)
        if U.size(weight="w") > 0:
            comms = nx.community.louvain_communities(U, weight="w", seed=seed)
            cmap = {n: i for i, c in enumerate(sorted(comms, key=len, reverse=True)) for n in c}
            table["community"] = [cmap[n] for n in nodes]
        else:  # no positive weights: modularity is undefined
            table["community"] = 0
    else:
        table["community"] = 0
    pos = nx.spring_layout(G, weight=wkey, seed=seed)
    metric = table[size_by].to_numpy(float)
    sizes = 150 + 850 * (metric - metric.min()) / (np.ptp(metric) or 1)
    ncomm = int(table["community"].max()) + 1
    cols = palette(ncomm)
    node_colors = [cols[c] for c in table["community"]]
    widths = np.ones(G.number_of_edges())
    if weighted:
        w = np.array([d[weight] for _, _, d in G.edges(data=True)], float)
        widths = 0.5 + 3.5 * (w - w.min()) / (np.ptp(w) or 1)
    info = {"graph": G, "positions": pos, "n_communities": ncomm, "betweenness_arithmetic": method,
            "betweenness_tolerance": tolerance, "duplicate_rows_merged": int(keep.sum()) - len(merged)}

    if interactive:
        go = require("plotly.graph_objects", "interactive")
        fig = go.Figure()
        marker_px = dict(zip(nodes, np.sqrt(sizes) * 1.3))
        for (u, v), wdt in zip(G.edges(), widths):
            if directed:  # an arrow annotation, stopping at the target marker's rim
                (x0, y0), (x1, y1) = pos[u], pos[v]
                if G.has_edge(v, u) and u != v:  # shift reciprocal arrows to their own side so both show
                    dx, dy = x1 - x0, y1 - y0
                    length = float(np.hypot(dx, dy)) or 1.0
                    ox, oy = 0.02 * dy / length, -0.02 * dx / length
                    x0, y0, x1, y1 = x0 + ox, y0 + oy, x1 + ox, y1 + oy
                fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y",
                                   axref="x", ayref="y", text="", showarrow=True, arrowhead=2, arrowsize=1,
                                   arrowwidth=float(wdt), arrowcolor="#aaaaaa", standoff=float(marker_px[v]) / 2)
            else:
                fig.add_trace(go.Scatter(x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]], mode="lines",
                                         line={"width": float(wdt), "color": "#aaaaaa"}, hoverinfo="skip",
                                         showlegend=False))
        hover = [f"<b>{r.node}</b><br>degree {r.degree}<br>strength {r.strength:.3g}<br>"
                 f"betweenness {r.betweenness:.3f}<br>community {r.community}" for r in table.itertuples()]
        fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
                                 mode="markers+text" if labels else "markers", text=[str(n) for n in nodes] if labels else None,
                                 textposition="top center", hovertext=hover, hoverinfo="text", showlegend=False,
                                 marker={"size": np.sqrt(sizes) * 1.3, "color": node_colors, "line": {"width": 1, "color": "white"}}))
        fig.update_layout(template="simple_white", xaxis={"visible": False}, yaxis={"visible": False},
                          margin={"l": 10, "r": 10, "t": 40, "b": 10},
                          title={"text": f"Node size: {size_by}; colour: community"})
        return VizResult(fig, None, table, info)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure
    if directed:  # arrows stop at the node rim; reciprocal pairs curve apart so both directions stay visible
        edges = list(G.edges())
        mutual = [G.has_edge(v, u) and u != v for u, v in edges]
        for curved in (False, True):
            chosen = [i for i, m in enumerate(mutual) if m == curved]
            if chosen:
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[edges[i] for i in chosen],
                                       width=[widths[i] for i in chosen], edge_color="#aaaaaa", arrows=True,
                                       arrowsize=14, arrowstyle="-|>", node_size=list(sizes),
                                       nodelist=nodes, connectionstyle="arc3,rad=0.15" if curved else "arc3")
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#aaaaaa", arrows=False)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=node_colors, edgecolors="white")
    if labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    if communities and ncomm > 1:
        ax.legend(handles=[Patch(color=c, label=f"community {i}") for i, c in enumerate(cols)], frameon=False,
                  loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title(f"Node size: {size_by}", loc="left", fontsize="medium", color=NEUTRAL)
    ax.axis("off")
    return VizResult(fig, ax, table, info)


def sankey(
    data: pd.DataFrame,
    source: str,
    target: str,
    value: str,
    title: str | None = None,
) -> VizResult:
    """Interactive Sankey diagram of flows between nodes (requires Plotly).

    Repeated source→target rows are summed. The table reports each node's
    total inflow and outflow.
    """
    go = require("plotly.graph_objects", "interactive")
    check_dataframe(data, [source, target, value])
    check_numeric(data, value)
    if (data[value] < 0).any():
        raise ValueError("flows must be non-negative")
    flows = data.groupby([source, target], sort=False, observed=True)[value].sum().reset_index()
    nodes = list(pd.unique(pd.concat([flows[source], flows[target]])))
    index = {n: i for i, n in enumerate(nodes)}
    cols = palette(len(nodes))
    fig = go.Figure(go.Sankey(
        node={"label": [str(n) for n in nodes], "pad": 15, "thickness": 18, "color": cols},
        link={"source": flows[source].map(index), "target": flows[target].map(index), "value": flows[value],
              "color": "rgba(150,150,150,0.35)"},
    ))
    fig.update_layout(title_text=title, font_size=12)
    inflow = flows.groupby(target)[value].sum()
    outflow = flows.groupby(source)[value].sum()
    table = pd.DataFrame({"node": nodes, "inflow": [inflow.get(n, 0) for n in nodes],
                          "outflow": [outflow.get(n, 0) for n in nodes]})
    return VizResult(fig, None, table, {"flows": flows})
