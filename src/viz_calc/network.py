"""Networks and flows (optional dependencies: ``networkx``, ``plotly``)."""

from __future__ import annotations

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

from ._core import (
    NEUTRAL,
    VizResult,
    check_choice,
    check_dataframe,
    check_distinct,
    check_has_values,
    check_numeric,
    cleanup_on_error,
    palette,
    require,
)

__all__ = ["network_map", "sankey"]


_PRIME = 2**127 - 1  # a Mersenne prime: path lengths are compared exactly through their residues modulo it


def _exact_weights(values: np.ndarray) -> list[Fraction] | None:
    """Exact values for weights that cannot have been rounded when stored, else ``None``.

    Integers qualify, and so do floats that print exactly as stored and could
    not be the rounding of another number with as few digits: integers below
    2**(mantissa bits + 1) (16,777,216 for float32; float32 16,800,009 is
    stored as 16,800,008) and fractions of at most the type's decimal
    precision (``0.125``, ``1001.5``). Other floats (``0.3``, ``73/9``,
    ``count/438``, float32 ``1000000.25``) may stand for a value the type
    cannot hold, and which one was meant is ambiguous, so the column uses the
    tolerant comparison instead.
    """
    out = []
    for v in values:
        if isinstance(v, (bool, np.bool_, int, np.integer)):
            out.append(Fraction(int(v)))
            continue
        if not isinstance(v, np.floating):
            return None
        info = np.finfo(v.dtype)
        text = np.format_float_positional(v, unique=True, trim="-")
        stored = Fraction(*v.as_integer_ratio())
        if stored != Fraction(text):
            return None
        if stored.denominator == 1:
            if abs(stored) >= 2 ** (info.nmant + 1):
                return None
        elif len(text.lstrip("-").replace(".", "").strip("0")) > info.precision:
            return None
        out.append(stored)
    return out


def _rounding_tolerance(dtype) -> float:
    """Relative rounding error of path lengths built from weights of this type: 4 machine epsilons.

    float32 and float16 hold only ~7 and ~3 significant digits, so a value
    entered and then rescaled can be off by a whole epsilon, and two equal
    paths by two; four leave a margin. The floor of 64 float64 epsilons covers
    the additions along a path.
    """
    eps = np.finfo(dtype).eps if np.dtype(dtype).kind == "f" else 0.0
    return max(64 * float(np.finfo(np.float64).eps), 4 * float(eps))


def _exact_distance(v, preds: dict, value: dict) -> Fraction:
    """Exact distance to v, summed along its first predecessors back to a node whose distance is known."""
    chain = []
    while v not in value:
        chain.append(v)
        v = preds[v][0][0]
    for u in reversed(chain):
        p, q = preds[u][0]
        value[u] = value[p] + q
    return value[chain[0]] if chain else value[v]


def _rational_shortest_paths(start: dict, into: dict) -> tuple[list, dict]:
    """Dijkstra in exact rational arithmetic from tentative distances *start*: nodes in order, and their distances.

    *into* maps each node to its in-edges ``(v, exact length)``.
    """
    out: dict = {}
    for w, edges in into.items():
        for v, q in edges:
            out.setdefault(v, []).append((w, q))
    counter = count()
    value, done, order = dict(start), set(), []
    heap = [(d, next(counter), v) for v, d in start.items()]
    heapq.heapify(heap)
    while heap:
        d, _, v = heapq.heappop(heap)
        if v in done:
            continue
        done.add(v)
        order.append(v)
        for w, q in out.get(v, []):
            if w not in value or d + q < value[w]:
                value[w] = d + q
                heapq.heappush(heap, (value[w], next(counter), w))
    return order, value


def _weighted_betweenness(G, distance: dict, rel_tol: float = 1e-10, rounding: float = 0.0,
                          exact: dict | None = None) -> dict:
    """Brandes' betweenness centrality with ties decided on the true shortest distance.

    ``distance[(u, v)]`` is the float length of edge u→v (for undirected
    graphs, both orientations are present); edges missing from it are not
    traversed. For each source, Dijkstra first finds the shortest distances,
    which do not depend on edge order. Then v is a predecessor of w on a
    shortest path when ``dist[v] < dist[w]`` and ``dist[v] + len(v, w)`` is
    tied with ``dist[w]``:

    * with *exact* (a dict of the same edges' exact :class:`~fractions.Fraction`
      lengths), a tie means exactly equal. Floats narrow the candidates to
      paths within 1e-11 of the shortest (more than float64 rounding over
      thousands of hops); their exact lengths are then compared through
      residues modulo the prime 2**127 - 1, and in rational arithmetic only
      when two candidates really differ. This costs about as much as the float
      version, unlike rational arithmetic throughout, whose denominators grow
      along every path. Where an edge is too short for float64 to separate its
      two ends, only the run of nodes whose distances floats cannot tell apart
      is ordered by Dijkstra in rational arithmetic.
    * otherwise, ``dist[v] + len(v, w)`` may exceed ``dist[w]`` by
      ``max(rounding, min(rel_tol, len(v, w) / (4 * dist[w]))) * dist[w]``:
      paths equal up to a relative *rel_tol* (1e-10, as in igraph) count as
      equally short (0.3, 73/9, count/total shares), but that slack never
      exceeds a quarter of the edge being tested, so a real extra hop is not a
      tie. Only differences within the weights' own *rounding* error, which
      low-precision floats cannot resolve, always tie. An edge too short for
      float64 to add to ``dist[v]`` leaves its ends at the same float
      distance; each such run of nodes is ordered by Dijkstra in rational
      arithmetic over those edges, from the nodes entered from outside the
      run, and the same rule on that scale picks the predecessors inside it.

    A path longer than the largest float raises :class:`ValueError`. The
    result does not depend on row order or on edges elsewhere. Normalized as
    in NetworkX: by 1/((n-1)(n-2)).

    References
    ----------
    Brandes, U. (2001). A faster algorithm for betweenness centrality.
    *Journal of Mathematical Sociology*, 25(2), 163–177.
    """
    nodes = list(G)
    out = {v: [(w, distance[(v, w)]) for w in G[v] if (v, w) in distance] for v in nodes}
    if exact is not None:
        into = {v: [] for v in nodes}  # w -> [(v, float length, exact length, residue)]
        for (v, w), q in exact.items():
            into[w].append((v, distance[(v, w)], q, q.numerator * pow(q.denominator, -1, _PRIME) % _PRIME))
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
        order = sorted(dist, key=dist.get)  # predecessors are closer (or ordered within a float-tied run below)
        if math.isinf(dist[order[-1]]):  # every overflowed length would tie with every other
            raise ValueError("a shortest path is longer than the largest float (its 1/weight edge lengths add up "
                             "past it); multiply the weights by a constant, which leaves betweenness unchanged")
        preds = {v: [] for v in dist}  # 2. predecessors on the shortest paths
        if exact is None:
            runs: dict = {}  # nodes at the same float distance, per distance
            for v in dist:
                runs.setdefault(dist[v], []).append(v)
                for w, length in out[v]:
                    # beyond the data's own rounding, the slack never exceeds a quarter of the edge being tested
                    slack = max(rounding * dist[w], min(rel_tol * dist[w], 0.25 * length))
                    if dist[v] < dist[w] and dist[v] + length <= dist[w] + slack:
                        preds[w].append(v)
            rank: dict = {}
            for run in runs.values():
                inside = {u: [] for u in run}
                members = set(run)
                for v in run:
                    for w, length in out[v]:
                        if w in members and dist[v] + length == dist[w]:  # too short to change the float sum
                            inside[w].append((v, Fraction(length)))
                if not any(inside.values()):
                    continue
                # an edge too short for float64 to add to the path length: floats cannot order its ends, so order
                # the run by exact distances from where it is entered, with the same tie rule on that scale
                start = {u: Fraction(0) for u in run if preds[u]}
                ranked, offset = _rational_shortest_paths(start, inside)
                for u in ranked:
                    for v, q in inside[u]:
                        slack = max(Fraction(rounding) * offset[u], min(Fraction(rel_tol) * offset[u], q / 4))
                        if v in offset and offset[v] < offset[u] and offset[v] + q <= offset[u] + slack:
                            preds[u].append(v)
                rank.update((u, i) for i, u in enumerate(ranked))
            if rank:
                order = sorted(dist, key=lambda v: (dist[v], rank.get(v, -1)))
        else:
            residue, value, exact_order, i = {s: 0}, {s: Fraction(0)}, [s], 1
            while i < len(order):
                w = order[i]
                window = dist[w] * (1 + 1e-11)
                if any(dist[v] >= dist[w] and dist[v] + length <= window for v, length, *_ in into[w] if v in dist):
                    # an edge too short for float64 to resolve against the path: floats cannot order its ends, so
                    # order this run of nodes that floats cannot separate in exact arithmetic
                    j = i + 1
                    while j < len(order) and dist[order[j]] <= dist[order[j - 1]] * (1 + 1e-11):
                        j += 1
                    run = set(order[i:j])
                    close = {u: [(v, q) for v, length, q, _ in into[u]
                                 if v in run or (v in residue and dist[v] + length <= dist[u] * (1 + 1e-11))]
                             for u in run}
                    start = {u: min(_exact_distance(v, preds, value) + q for v, q in edges if v not in run)
                             for u, edges in close.items() if any(v not in run for v, _ in edges)}
                    inside = {u: [(v, q) for v, q in edges if v in run] for u, edges in close.items()}
                    ranked, exact_value = _rational_shortest_paths(start, inside)
                    value.update(exact_value)
                    for u in ranked:
                        preds[u] = [(v, q) for v, q in close[u] if value[v] + q == value[u]]
                        residue[u] = value[u].numerator * pow(value[u].denominator, -1, _PRIME) % _PRIME
                    exact_order += ranked
                    i = j
                    continue
                near = [(v, q, (residue[v] + r) % _PRIME) for v, length, q, r in into[w]
                        if v in dist and dist[v] < dist[w] and dist[v] + length <= window]
                if len({r for *_, r in near}) > 1:  # candidates that really differ: keep the exactly shortest
                    totals = [_exact_distance(v, preds, value) + q for v, q, _ in near]
                    value[w] = min(totals)
                    near = [c for c, t in zip(near, totals) if t == value[w]]
                preds[w] = [(v, q) for v, q, _ in near]
                residue[w] = near[0][2]
                exact_order.append(w)
                i += 1
            order = exact_order
            preds = {w: [v for v, _ in ps] for w, ps in preds.items()}
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


@cleanup_on_error
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
    implementation. Node size encodes *size_by* (``"degree"``, ``"strength"``
    or ``"betweenness"``); colour encodes community.
    The layout is seeded so it is reproducible. Rows missing *source* or
    *target* are left out, whatever their weight. Repeated edges (including
    B→A after A→B in an undirected graph) are merged, summing their weights;
    missing weights raise an error. Weighted betweenness uses 1/weight as edge
    length. Weights that cannot have been rounded when stored (integers, or
    floats such as 0.5 or 1001.25 that print exactly as stored) are compared
    exactly, so equally short paths always tie. Other floats (0.3, 73/9,
    shares such as count/438) may stand for a value the type cannot hold, so
    path lengths within a relative 1e-10 of each other count as equal, as in
    igraph, unless the difference is more than a quarter of a path's last
    edge (a real extra hop). float32 and float16 columns hold only ~7 and ~3
    digits, so there any difference within four machine epsilons (4.8e-7 and
    0.4%) counts as equal. The choice is made for the whole weight column,
    and the result does not depend on row order. If a shortest path's length
    (the sum of 1/weight along it) exceeds the largest float, a
    ``ValueError`` asks for the weights to be rescaled, as does a node
    strength above the largest float. Weights above 2**256 are divided by a
    power of two for community detection and the layout, which would
    otherwise overflow; this is exact and does not change modularity.
    ``info["betweenness_arithmetic"]`` says which was used and
    ``info["betweenness_tolerance"]`` gives the tolerance. For directed
    graphs, communities are found on the undirected graph with reciprocal
    weights summed.

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
    check_distinct(source=source, target=target, weight=weight)
    check_choice("size_by", size_by, ["degree", "strength", "betweenness"])
    weighted = weight is not None  # a column may legitimately be labelled 0
    if weighted:
        check_numeric(data, weight)
    check_has_values(data, source, target, weight)
    # Work on plain Python objects: categorical columns, non-string labels and nullable dtypes all behave the same.
    src = data[source].to_numpy(dtype=object)
    tgt = data[target].to_numpy(dtype=object)
    keep = ~(pd.isna(src) | pd.isna(tgt))
    if not keep.any():
        raise ValueError(f"no row has both a {source!r} and a {target!r} value, so there are no edges")
    wts = data[weight].to_numpy(dtype=float, na_value=np.nan) if weighted else np.ones(len(data))
    if weighted and np.isnan(wts[keep]).any():
        raise ValueError(f"edge weights in {weight!r} contain missing values; drop or fill them first")
    if weighted and np.isinf(wts[keep]).any():
        raise ValueError(f"edge weights in {weight!r} must be finite")
    if weighted and (wts[keep] < 0).any():
        raise ValueError("edge weights must be non-negative")
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
            try:
                total = sum(ws, Fraction(0)) if exact_values is not None else math.fsum(ws)
                G[u][v][weight] = float(total)
            except OverflowError:
                raise ValueError(f"the {weight!r} weights of the repeated edge {u!r}-{v!r} add up to more than the "
                                 "largest float") from None
            edge_weight[(u, v)] = total
    wkey = weight if weighted else None
    nodes = list(G.nodes())
    table = pd.DataFrame({"node": nodes})
    table["degree"] = [G.degree(n) for n in nodes]
    table["strength"] = [G.degree(n, weight=wkey) for n in nodes]
    if weighted and np.isinf(table["strength"]).any():
        node = table["node"][np.isinf(table["strength"])].iloc[0]
        raise ValueError(f"the {weight!r} weights at node {node!r} add up to more than the largest float; divide "
                         "the weights by a constant")
    # Louvain squares sums of strengths and the spring layout multiplies weights, so both overflow long before
    # the weights do. Very large weights are divided by a power of two for them: exact, modularity is unchanged,
    # and the layout no longer changes with the weights' scale at that size.
    exponent = math.frexp(max(G[u][v][weight] for u, v in G.edges()))[1] if weighted else 0
    scale = math.ldexp(1.0, 256 - exponent) if exponent > 256 else 1.0
    # Betweenness treats weights as distances; stronger ties should be shorter, so use 1/weight; zero-weight
    # edges carry no tie and are left out. Weights stored exactly are compared exactly, so equally short paths
    # always tie; other floats, whose intended exact value cannot be known, use a tolerance that absorbs rounding
    # but never a real extra hop.
    method = tolerance = None
    if weighted:
        distance, lengths = {}, {}
        for (u, v), w in edge_weight.items():
            if w > 0 and u != v:
                d = 1.0 / float(w)
                if math.isinf(d):
                    raise ValueError(f"edge weight {float(w)!r} in {weight!r} is too small to use as a distance "
                                     "(1/weight overflows)")
                q = 1 / w if exact_values is not None else None
                for e in [(u, v)] if directed else [(u, v), (v, u)]:
                    distance[e], lengths[e] = d, q
        if exact_values is not None and all(q.denominator % _PRIME for q in lengths.values()):
            method = "exact"
            bc = _weighted_betweenness(G, distance, exact=lengths)
        else:
            method = "tolerant floating-point"
            rounding = _rounding_tolerance(raw.dtype)
            tolerance = max(1e-10, rounding)
            bc = _weighted_betweenness(G, distance, rounding=rounding)
    else:
        bc = nx.betweenness_centrality(G)
    table["betweenness"] = [bc[n] for n in nodes]
    if communities:
        # Communities are found on the undirected graph; reciprocal edges A→B and B→A add their weights.
        U = nx.Graph()
        U.add_nodes_from(G)
        for u, v, d in G.edges(data=True):
            w = d[weight] * scale if weighted else 1.0
            if U.has_edge(u, v):
                U[u][v]["w"] += w
            else:
                U.add_edge(u, v, w=w)
        if U.size(weight="w") > 0:
            comms = nx.community.louvain_communities(U, weight="w", seed=None if seed is None else int(seed))
            cmap = {n: i for i, c in enumerate(sorted(comms, key=len, reverse=True)) for n in c}
            table["community"] = [cmap[n] for n in nodes]
        else:  # no positive weights: modularity is undefined
            table["community"] = 0
    else:
        table["community"] = 0
    layout = G
    if scale != 1.0:
        layout = G.copy()
        for _, _, d in layout.edges(data=True):
            d[weight] *= scale
    pos = nx.spring_layout(layout, weight=wkey, seed=None if seed is None else int(seed))
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


@cleanup_on_error
def sankey(
    data: pd.DataFrame,
    source: str,
    target: str,
    value: str,
    title: str | None = None,
) -> VizResult:
    """Interactive Sankey diagram of flows between nodes (requires Plotly).

    Repeated source→target rows are summed; a missing source, target or flow
    value, or a negative flow, raises an error (dropping the row would
    silently change the totals). The table reports each node's total inflow
    and outflow.
    """
    go = require("plotly.graph_objects", "interactive")
    check_dataframe(data, [source, target, value])
    check_numeric(data, value)
    check_distinct(source=source, target=target, value=value)
    check_has_values(data, source, target, value)
    for column in (source, target):
        missing = int(data[column].isna().sum())
        if missing:
            raise ValueError(f"{missing} row(s) have no {column!r} value; drop or fill them first")
    if data[value].isna().any():
        raise ValueError(f"flow values in {value!r} contain missing values; drop or fill them first")
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
