import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def create_heatmap(data, variables, interactive=False, aesthetics=None, facet_by=None, title=None, bins=10):
    """
    Create a heatmap.

    Parameters:
    data (pd.DataFrame): The data to plot
    variables (list of str): The variables to plot (should be 2 variables)
    interactive (bool, optional): Use interactive plotly plot if True, else seaborn plot
    aesthetics (dict, optional): Additional aesthetic parameters
    facet_by (str, optional): Column name to facet by
    title (str, optional): Plot title
    bins (int, optional): Number of bins along each axis

    Returns:
    None
    """
    aesthetics = aesthetics or {}
    
    data['bin_x'] = pd.cut(data[variables[0]], bins=bins).astype(str)
    data['bin_y'] = pd.cut(data[variables[1]], bins=bins).astype(str)
    heatmap_data = data.groupby(['bin_x', 'bin_y', facet_by]).size().reset_index(name='count')
    
    if interactive:
        aesthetics.setdefault("color_continuous_scale", "Viridis")
        if facet_by:
            for facet in data[facet_by].unique():
                subset_data = heatmap_data[heatmap_data[facet_by] == facet]
                fig = px.density_heatmap(subset_data, x='bin_x', y='bin_y', z='count', title=f"{title} - {facet}", **aesthetics)
                fig.show()
        else:
            fig = px.density_heatmap(heatmap_data, x='bin_x', y='bin_y', z='count', title=title, **aesthetics)
            fig.show()
    else:
        aesthetics.setdefault("cmap", "viridis")
        if facet_by:
            for facet in data[facet_by].unique():
                subset_data = heatmap_data[heatmap_data[facet_by] == facet]
                subset_pivot = subset_data.pivot('bin_y', 'bin_x', 'count')
                sns.heatmap(subset_pivot, **aesthetics)
                plt.title(f"{title} - {facet}")
                plt.show()
        else:
            pivot_data = heatmap_data.pivot('bin_y', 'bin_x', 'count')
            sns.heatmap(pivot_data, **aesthetics)
            if title:
                plt.title(title)
            plt.show()

# Creating synthetic data
np.random.seed(42)
data_size = 1000
data = pd.DataFrame({
    'variable1': np.random.randn(data_size),
    'variable2': np.random.rand(data_size),
    'category': np.random.choice(['A', 'B', 'C'], size=data_size)
})

# Usage
create_heatmap(data, ['variable1', 'variable2'], interactive=True, facet_by='category', title="Interactive Heatmap")
