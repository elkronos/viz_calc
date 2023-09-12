import plotly.graph_objects as go
import pandas as pd

def create_sankeydiagram(data, source_col, target_col, value_col, title=None, font_size=10, font_color='black'):
    """
    Create a Sankey diagram with optional parameters for customization.

    Parameters:
    data (pd.DataFrame): Input data frame
    source_col (str): Column name of the source variable
    target_col (str): Column name of the target variable
    value_col (str): Column name of the value variable
    title (str, optional): Title of the plot. Defaults to None.
    font_size (int, optional): Font size for labels. Defaults to 10.
    font_color (str, optional): Font color for labels. Defaults to 'black'.

    Returns:
    None
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    if source_col not in data.columns or target_col not in data.columns or value_col not in data.columns:
        raise ValueError("Column names not found in data")

    # Mapping unique labels to integers
    unique_labels = pd.concat([data[source_col], data[target_col]]).unique()
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    # Constructing sources, targets and values for the sankey diagram
    sources = data[source_col].map(label_to_int)
    targets = data[target_col].map(label_to_int)
    values = data[value_col]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=unique_labels,
            color='blue'  # setting a unified color for nodes, you can customize this
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color='rgba(217, 217, 217, 0.5)'  # setting a soft grey color for links, you can customize this
        )
    ))

    if title:
        fig.update_layout(title_text=title, font_size=font_size, font_color=font_color)

    fig.show()

# Example usage:

data = pd.DataFrame({
    'Source': ['A', 'B', 'A', 'C', 'D'],
    'Target': ['C', 'D', 'D', 'D', 'E'],
    'Value': [10, 15, 10, 10, 20]
})

create_sankeydiagram(data, 'Source', 'Target', 'Value', title='Sankey Diagram Example', font_size=12, font_color='darkblue')
