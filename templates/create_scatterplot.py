import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Union

def create_scatterplot(data: pd.DataFrame, x_col: str, y_col: str, category_col: str, 
                       group_col: Optional[str] = None, symbol_col: Optional[str] = None, 
                       title: str = 'Scatter Plot', template: str = 'plotly_dark', 
                       trendline: Optional[str] = None, marker_size: int = 5, 
                       marker_opacity: float = 0.7, x_label: Optional[str] = None, 
                       y_label: Optional[str] = None, hover_data: Optional[List[str]] = None, 
                       legend_title: str = 'Category', title_font: Optional[Dict[str, Union[str, int]]] = None, 
                       axis_label_font: Optional[Dict[str, Union[str, int]]] = None, 
                       annotations: Optional[List[Dict[str, Union[str, int]]]] = None, 
                       interactive: bool = False) -> None:
    """
    Create a scatter plot with optional interactivity and customization.

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
    - interactive (bool, optional): Whether to use an interactive Plotly plot.
    
    Returns:
    None
    """
    if x_col not in data.columns or y_col not in data.columns or category_col not in data.columns:
        raise ValueError("x_col, y_col, or category_col does not exist in the DataFrame.")
    
    if group_col and group_col not in data.columns:
        raise ValueError(f"{group_col} does not exist in the DataFrame.")
    
    if symbol_col and symbol_col not in data.columns:
        raise ValueError(f"{symbol_col} does not exist in the DataFrame.")
    
    if interactive:
        import plotly.express as px

        fig = px.scatter(data, x=x_col, y=y_col, color=category_col, facet_col=group_col, 
                         trendline=trendline, template=template, title=title)
        fig.update_traces(marker=dict(size=marker_size, opacity=marker_opacity))

        if x_label:
            fig.update_xaxes(title_text=x_label)

        if y_label:
            fig.update_yaxes(title_text=y_label)

        if symbol_col:
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
        
    else:
        plt.figure(figsize=(10,6))
        categories = data[category_col].unique()
        for category in categories:
            subset = data[data[category_col] == category]
            plt.scatter(subset[x_col], subset[y_col], label=category, s=marker_size*10, alpha=marker_opacity)
        
        plt.title(title)
        plt.xlabel(x_label if x_label else x_col)
        plt.ylabel(y_label if y_label else y_col)
        plt.legend(title=legend_title)
        plt.show()


# Example DataFrame
data = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'C', 'B'],
    'X_value': [1.2, 2.4, 3.1, 4.7, 2.9],
    'Y_value': [5.6, 4.3, 3.6, 1.8, 4.1],
    'Group': ['X', 'Y', 'Z', 'X', 'Z'],
    'Symbol': ['circle', 'square', 'diamond', 'circle', 'square']
})

# Using the function to create a non-interactive scatter plot
create_scatterplot(data, 'X_value', 'Y_value', 'Category', interactive=False)

# Example DataFrame 1 with multiple groups and symbols
data1 = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'],
    'X_value': [1.2, 2.4, 3.1, 4.7, 2.9, 3.5, 4.1, 1.8],
    'Y_value': [5.6, 4.3, 3.6, 1.8, 4.1, 2.7, 3.2, 5.0],
    'Group': ['X', 'Y', 'Z', 'X', 'Z', 'Y', 'X', 'Z'],
    'Symbol': ['circle', 'square', 'diamond', 'circle', 'diamond', 'square', 'circle', 'square']
})

# Function call with interactive=True
create_scatterplot(data1, 'X_value', 'Y_value', 'Category', group_col='Group', symbol_col='Symbol', interactive=True)


# Example DataFrame 2 without group_col and symbol_col
data2 = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'A'],
    'X_value': [1.5, 3.0, 4.2, 2.4],
    'Y_value': [4.6, 3.8, 2.4, 4.0],
})

# Function call with default settings
create_scatterplot(data2, 'X_value', 'Y_value', 'Category', interactive=True)
