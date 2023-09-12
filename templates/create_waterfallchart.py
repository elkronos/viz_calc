import plotly.graph_objects as go
import pandas as pd

def create_waterfallchart(data, values_col, names_col, title=None, interactive=True):
    """
    Create a waterfall chart with optional parameters for customization.

    Parameters:
    data (pd.DataFrame): Input data frame
    values_col (str): Column name of the values
    names_col (str): Column name of the segment names
    title (str, optional): Title of the plot. Defaults to None.
    interactive (bool, optional): Whether to create an interactive plot. Defaults to True.

    Returns:
    None
    """
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")

    if values_col not in data.columns or names_col not in data.columns:
        raise ValueError("Column names not found in data")

    if data.shape[0] < 2:
        raise ValueError("Data must contain at least two rows")

    # Calculating measures for the waterfall chart
    measures = [data.loc[0, values_col]]
    measures.extend(data[values_col].diff().tolist()[1:])
    
    # Creating waterfall chart
    fig = go.Figure(go.Waterfall(
        x = data[names_col],
        y = measures,
        textposition = "outside",
        text = data[names_col]
    ))

    if title:
        fig.update_layout(title_text=title)
    
    if interactive:
        fig.show()
    else:
        fig.show("png")

# Generating synthetic data
data = pd.DataFrame({
    'Names': ['Start', 'Sale 1', 'Refund 1', 'Sale 2', 'Sale 3', 'Refund 2', 'Sale 4', 'End'],
    'Values': [1000, 1200, 1100, 1300, 1500, 1400, 1600, 1700]
})

# Using the function to create a waterfall chart
create_waterfallchart(data, 'Values', 'Names', title='Waterfall Chart Example', interactive=True)
