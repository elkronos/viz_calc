import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np

def create_boxplot(data, numeric_col, categorical_col, group_col=None, colors=None, title=None, interactive=False):
    """
    Create a boxplot with optional parameters for customization.

    Parameters:
    data (pd.DataFrame): Input data frame
    numeric_col (str): Column name of the numeric variable
    categorical_col (str): Column name of the categorical variable
    group_col (str, optional): Column name to group data by for faceted boxplots. Defaults to None.
    colors (str or list, optional): Colors for the boxplots. Defaults to None.
    title (str, optional): Title of the plot. Defaults to None.
    interactive (bool, optional): Whether to create an interactive plot. Defaults to False.

    Returns:
    None
    """
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")

    if numeric_col not in data.columns or categorical_col not in data.columns or (group_col and group_col not in data.columns):
        raise ValueError("Column names not found in data")

    if interactive:
        fig = px.box(data, x=categorical_col, y=numeric_col, color=group_col, title=title, template="plotly_dark" if colors == "dark" else None)
        fig.show()
    else:
        sns.set_theme(style="darkgrid" if colors == "dark" else "whitegrid")
        if group_col:
            g = sns.catplot(x=categorical_col, y=numeric_col, data=data, col=group_col, kind="box")
            if title:
                g.fig.suptitle(title)
        else:
            ax = sns.boxplot(x=data[categorical_col], y=data[numeric_col])
            if title:
                ax.set_title(title)
        plt.show()

# Generating synthetic data
np.random.seed(42)
data_size = 100
categories = ['A', 'B', 'C', 'D']
groups = ['X', 'Y']

data = pd.DataFrame({
    'Numeric': np.random.randn(data_size),
    'Categorical': np.random.choice(categories, data_size),
    'Group': np.random.choice(groups, data_size)
})

# Using the function to create a boxplot
create_boxplot(data, 'Numeric', 'Categorical', group_col='Group', colors='dark', title='Boxplot Example', interactive=True)
