import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

def create_histogram(data, variable, binwidth=None, interactive=False, aesthetics=None, facet_by=None, title=None, facet_order='ascending'):
    """
    Create a histogram.

    Parameters:
    data (pd.DataFrame): The data to plot.
    variable (str): The variable to plot.
    binwidth (float, optional): Binwidth; if not provided, it will be computed using Freedman-Diaconis rule.
    interactive (bool, optional): Use interactive plotly plot if True, else seaborn plot. Default is False.
    aesthetics (dict, optional): Additional aesthetic parameters for the seaborn plot.
    facet_by (str, optional): Column name to facet by.
    title (str, optional): Plot title.
    facet_order (str, optional): Order in which to display the facets ('ascending', 'descending'). Default is 'ascending'.

    Returns:
    None
    """
    
    aesthetics = aesthetics or {}

    if binwidth is None:
        iqr = np.subtract(*np.percentile(data[variable], [75, 25]))
        binwidth = 2 * iqr / (len(data[variable]) ** (1/3))

    if facet_order not in ['ascending', 'descending']:
        raise ValueError("Invalid facet_order. Choose from 'ascending' or 'descending'.")

    category_order = sorted(data[facet_by].unique(), reverse=(facet_order=='descending'))

    if interactive:
        fig = px.histogram(data, x=variable, nbins=int(binwidth), facet_row=facet_by, 
                           category_orders={facet_by: category_order}, 
                           title=title, **aesthetics)
        fig.update_layout(bargap=0.1)
        fig.update_yaxes(title_text='Frequency')
        fig.update_xaxes(title_text=variable)
        fig.show()
    else:
        g = sns.FacetGrid(data, col=facet_by, sharey=False, col_wrap=2, col_order=category_order)
        g.map(sns.histplot, variable, binwidth=binwidth, **aesthetics)
        
        # Adding labels and titles
        g.set_axis_labels(variable, "Frequency")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")

        if title:
            plt.subplots_adjust(top=0.9)
            g.fig.suptitle(title)

        plt.show()

# Creating synthetic data
np.random.seed(42)
data_size = 1000
data = pd.DataFrame({
    'variable1': np.random.randn(data_size),
    'variable2': np.random.rand(data_size),
    'category': np.random.choice(['A', 'B', 'C', 'D'], size=data_size)
})

# Usage
create_histogram(data, 'variable1', interactive=True, facet_by='category', title="Histogram Analysis by Category", facet_order='descending')
