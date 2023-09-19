import networkx as nx
import plotly.graph_objects as go
import pandas as pd
from community import community_louvain

def network_map(data, source_col, target_col, edge_weight_col=None, node_size_col=None, node_color_col=None, colorscale='Viridis', interactive=False, community_detection=False):
    """
    Function to create a network map from any dataset.
    
    Parameters:
    data (pd.DataFrame): Input dataset containing the network information.
    source_col (str): Column name representing the source nodes in the dataset.
    target_col (str): Column name representing the target nodes in the dataset.
    edge_weight_col (str): Column name representing the edge weights in the dataset. Default is None.
    node_size_col (str): Column name representing the node sizes in the dataset. Default is None.
    node_color_col (str): Column name representing the node colors in the dataset. Default is None.
    colorscale (str): Colorscale to use for the node colors. Default is 'Viridis'.
    interactive (bool): Whether to create an interactive network map. Default is False.
    community_detection (bool): Whether to apply community detection algorithm to find communities in the network. Default is False.
    
    Returns:
    None
    """
    G = nx.from_pandas_edgelist(data, source=source_col, target=target_col, edge_attr=True)
    
    if community_detection:
        community_dict = community_louvain.best_partition(G)
        nx.set_node_attributes(G, community_dict, 'community')
    else:
        community_dict = {node: 0 for node in G.nodes()}
    
    pos = nx.spring_layout(G)
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    node_sizes = nx.get_node_attributes(G, node_size_col) if node_size_col else {node: 10 for node in G.nodes()}
    node_colors = nx.get_node_attributes(G, node_color_col) if node_color_col else {node: 1 for node in G.nodes()}
    
    edge_traces = [
        go.Scatter(
            x=[x0, x1], 
            y=[y0, y1],
            line=dict(width=edge[2].get(edge_weight_col, 1) if edge_weight_col else 1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        for edge in G.edges(data=True)
        for x0, y0 in [pos[edge[0]]]
        for x1, y1 in [pos[edge[1]]]
    ]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale=colorscale,
            color=list(node_colors.values()),
            size=list(node_sizes.values()),
            colorbar=dict(
                thickness=15,
                title='Node Metrics',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)
        ),
        text=[f'Node: {node}<br>Community: {community_dict[node]}' for node in G.nodes()],
        textposition="top center"
    )

    fig = go.Figure(data=[*edge_traces, node_trace],
                    layout=go.Layout(
                        title='Network Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        annotations=[
                            dict(
                                text="Python code: <a href='https://www.plotly.com/'> Plotly</a>",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002
                            )
                        ]
                    )
                )
    fig.show()
    
# Synthetic Example representing a small social network
data = pd.DataFrame({
    'Source': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve', 'Frank', 'Grace', 'Heidi'],
    'Target': ['Bob', 'Charlie', 'Dave', 'Eve', 'Frank', 'Grace', 'Heidi', 'Alice'],
    'Friendship Strength': [7, 5, 3, 8, 2, 6, 4, 7],
    'Interactions Frequency': [10, 8, 6, 9, 3, 7, 5, 8],
    'Source Age': [25, 30, 35, 40, 45, 50, 55, 60],
    'Target Age': [30, 35, 40, 45, 50, 55, 60, 25],
    'Source Group': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
    'Target Group': ['A', 'B', 'B', 'C', 'C', 'D', 'D', 'A']
})

# Function call with updated parameters
network_map(
    data, 
    'Source', 
    'Target', 
    edge_weight_col='Friendship Strength', 
    node_size_col='Source Age', 
    node_color_col='Interactions Frequency', 
    colorscale='Viridis', 
    interactive=True, 
    community_detection=True
)
