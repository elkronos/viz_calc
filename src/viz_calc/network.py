"""Networks and flows (optional dependencies: ``networkx``, ``plotly``)."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from ._core import NEUTRAL, VizResult, check_dataframe, check_numeric, palette, require

__all__ = ["network_map", "sankey"]


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
    missing weights raise an error. For directed graphs, communities are found
    on the undirected graph with reciprocal weights summed.

    Set ``interactive=True`` for a Plotly figure with hover details.

    References
    ----------
    Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008).
    Fast unfolding of communities in large networks. *J. Stat. Mech.*, P10008.
    """
    nx = require("networkx", "network")
    check_dataframe(data, [source, target, weight])
    if weight:
        check_numeric(data, weight)
        if (data[weight] < 0).any():
            raise ValueError("edge weights must be non-negative")
    # Work on plain Python objects: categorical columns, non-string labels and nullable dtypes all behave the same.
    src = data[source].to_numpy(dtype=object)
    tgt = data[target].to_numpy(dtype=object)
    keep = ~(pd.isna(src) | pd.isna(tgt))
    wts = data[weight].to_numpy(dtype=float, na_value=np.nan) if weight else np.ones(len(data))
    if weight and np.isnan(wts[keep]).any():
        raise ValueError(f"edge weights in {weight!r} contain missing values; drop or fill them first")
    # Repeated edges are combined (weights summed) instead of letting the last row win; each edge keeps
    # the orientation it was first seen with.
    merged: dict = {}
    for u, v, w in zip(src[keep], tgt[keep], wts[keep]):
        key = (u, v) if directed else frozenset((u, v))
        if key in merged:
            merged[key][2] += w
        else:
            merged[key] = [u, v, w]
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(pd.unique(np.column_stack([src[keep], tgt[keep]]).ravel()))  # first-appearance order
    for u, v, w in merged.values():
        G.add_edge(u, v)
        if weight:
            G[u][v][weight] = float(w)
    wkey = weight if weight else None
    nodes = list(G.nodes())
    table = pd.DataFrame({"node": nodes})
    table["degree"] = [G.degree(n) for n in nodes]
    table["strength"] = [G.degree(n, weight=wkey) for n in nodes]
    # Betweenness treats weights as distances; stronger ties should be shorter, so use 1/weight.
    # Zero-weight edges carry no tie and are left out of the paths.
    if weight:
        H = G.__class__()
        H.add_nodes_from(G)
        H.add_edges_from((u, v, {"distance": 1.0 / d[weight]}) for u, v, d in G.edges(data=True) if d[weight] > 0)
        bc = nx.betweenness_centrality(H, weight="distance")
    else:
        bc = nx.betweenness_centrality(G)
    table["betweenness"] = [bc[n] for n in nodes]
    if communities:
        # Communities are found on the undirected graph; reciprocal edges A→B and B→A add their weights.
        U = nx.Graph()
        U.add_nodes_from(G)
        for u, v, d in G.edges(data=True):
            w = d[weight] if weight else 1.0
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
    if weight:
        w = np.array([d[weight] for _, _, d in G.edges(data=True)], float)
        widths = 0.5 + 3.5 * (w - w.min()) / (np.ptp(w) or 1)
    info = {"graph": G, "positions": pos, "n_communities": ncomm,
            "duplicate_rows_merged": int(keep.sum()) - len(merged)}

    if interactive:
        go = require("plotly.graph_objects", "interactive")
        fig = go.Figure()
        for (u, v), wdt in zip(G.edges(), widths):
            fig.add_trace(go.Scatter(x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]], mode="lines",
                                     line={"width": float(wdt), "color": "#aaaaaa"}, hoverinfo="skip", showlegend=False))
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
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#aaaaaa", arrows=directed)
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
