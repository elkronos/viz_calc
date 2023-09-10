import pandas as pd
import numpy as np
import plotly.express as px

def generate_synthetic_data(data_size=100):
    """Generates synthetic data for demonstration purposes."""
    np.random.seed(42)
    categories = ['A', 'B', 'C']
    groups = ['X', 'Y', 'Z']
    
    data = pd.DataFrame({
        'Category': np.random.choice(categories, data_size),
        'Metric1': np.random.randn(data_size),
        'Metric2': np.random.randn(data_size),
        'Metric3': np.random.randn(data_size),
        'Metric4': np.random.randn(data_size),
        'Group': np.random.choice(groups, data_size)
    })
    
    return data

def create_radarplot(data: pd.DataFrame, metrics: list, categories_col: str, title: str = 'Radar Plot', template: str = 'plotly_dark', radial_range: tuple = (None, None)):
    """
    Create an interactive radar plot with optional parameters for customization.
    
    Parameters:
    data (pd.DataFrame): Input data frame.
    metrics (list): List of column names representing the metrics.
    categories_col (str): Column name representing the categories.
    title (str, optional): Title of the plot. Defaults to 'Radar Plot'.
    template (str, optional): Template for the plot style. Defaults to 'plotly_dark'.
    radial_range (tuple, optional): The range of the radial axes. Defaults to (None, None), which automatically determines the range.
    
    Returns:
    None
    """
    
    valid_templates = ['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white', 'none']
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    if template not in valid_templates:
        raise ValueError(f"Invalid template. Valid options are: {', '.join(valid_templates)}")

    missing_cols = [col for col in metrics + [categories_col] if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Column names not found in data: {', '.join(missing_cols)}")

    data_melted = data.melt(id_vars=[categories_col], value_vars=metrics)
    
    fig = px.line_polar(data_melted, r='value', theta='variable', color=categories_col, line_close=True, title=title, template=template, range_r=radial_range)
    fig.update_traces(fill='toself')
    fig.show()

# Using the function to create a radar plot
data = generate_synthetic_data()
create_radarplot(data, ['Metric1', 'Metric2', 'Metric3', 'Metric4'], 'Category')
