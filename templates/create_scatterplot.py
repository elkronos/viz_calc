import pandas as pd
import numpy as np
import plotly.express as px

def generate_synthetic_data(data_size: int = 100) -> pd.DataFrame:
    """Generates synthetic data for demonstration purposes."""
    np.random.seed(42)
    categories = ['A', 'B', 'C']
    groups = ['X', 'Y', 'Z']

    data = pd.DataFrame({
        'Category': np.random.choice(categories, data_size),
        'X_value': np.random.randn(data_size),
        'Y_value': np.random.randn(data_size),
        'Group': np.random.choice(groups, data_size),
        'Symbol': np.random.choice(['circle', 'square', 'diamond'], data_size)
    })

    return data

def create_scatterplot(data: pd.DataFrame, x_col: str, y_col: str, category_col: str, group_col: str = None, symbol_col: str = None, title: str = 'Interactive Scatter Plot', template: str = 'plotly_dark', trendline: str = None, marker_size: int = 5, marker_opacity: float = 0.7, x_label: str = None, y_label: str = None, hover_data: list = None, legend_title: str = 'Category', title_font: dict = None, axis_label_font: dict = None, annotations: list = None):
    """
    Create an interactive scatter plot with optional parameters for customization.

    Parameters:
    - data (pd.DataFrame): The input data frame.
    - x_col (str): Column name representing the x-values.
    - y_col (str): Column name representing the y-values.
    - category_col (str): Column name representing the categories.
    - group_col (str, optional): Column name for grouping data (creates faceted plots).
    - symbol_col (str, optional): Column name dictating the symbol used for each data point.
    - title (str, optional): Title of the plot.
    - template (str, optional): Template for the plot style.
    - trendline (str, optional): Type of trendline ('ols' for ordinary least squares).
    - marker_size (int, optional): Size of the markers.
    - marker_opacity (float, optional): Opacity of the markers.
    - x_label (str, optional): Label for the x-axis.
    - y_label (str, optional): Label for the y-axis.
    - hover_data (list, optional): Additional data to display on hover.
    - legend_title (str, optional): Title for the legend.
    - title_font (dict, optional): Font properties for the title.
    - axis_label_font (dict, optional): Font properties for the axis labels.
    - annotations (list, optional): List of annotations to add to the plot.
    
    Returns:
    None
    """
    fig = px.scatter(data, x=x_col, y=y_col, color=category_col, facet_col=group_col, trendline=trendline, template=template, title=title)
    fig.update_traces(marker=dict(size=marker_size, opacity=marker_opacity))

    if x_label:
        fig.update_xaxes(title_text=x_label)
    
    if y_label:
        fig.update_yaxes(title_text=y_label)
    
    if symbol_col and symbol_col in data.columns:
        fig.update_traces(marker_symbol=data[symbol_col])

    if legend_title:
        fig.update_layout(legend_title=dict(text=legend_title))
    
    if title_font:
        fig.update_layout(title_font=title_font)

    if axis_label_font:
        fig.update_xaxes(title_font=axis_label_font)
        fig.update_yaxes(title_font=axis_label_font)

    if annotations:
        for annotation in annotations:
            fig.add_annotation(annotation)
    
    fig.show()

# Using the function to create an interactive scatter plot
data = generate_synthetic_data()
create_scatterplot(data, 'X_value', 'Y_value', 'Category', group_col='Group', symbol_col='Symbol', hover_data=['Group'], legend_title='Category Legend', title_font={'size': 24, 'family': 'Courier New, monospace', 'color': 'RebeccaPurple'}, axis_label_font={'size': 14, 'family': 'Courier New, monospace'}, annotations=[dict(x=0, y=0, xref='x', yref='y', text='Origin', showarrow=True, arrowhead=2, ax=20, ay=-30)])
