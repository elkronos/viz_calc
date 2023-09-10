import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def create_histogram(data, variable, binwidth=None, interactive=False, aesthetics=None, facet_by=None, title=None):
    """
    Create a histogram.

    Parameters:
    data (pd.DataFrame): The data to plot
    variable (str): The variable to plot
    binwidth (float, optional): Binwidth; computes optimally if not provided
    interactive (bool, optional): Use interactive plotly plot if True, else matplotlib plot
    aesthetics (dict, optional): Additional aesthetic parameters
    facet_by (str, optional): Column name to facet by
    title (str, optional): Plot title

    Returns:
    None
    """

    aesthetics = aesthetics or {}

    # Compute optimal binwidth using Freedman-Diaconis rule if not provided
    if binwidth is None:
        iqr = np.subtract(*np.percentile(data[variable], [75, 25]))
        binwidth = 2 * iqr * (len(data[variable]) ** (-1/3))

    if interactive:
        aesthetics.setdefault("color", facet_by)
        fig = px.histogram(data, x=variable, nbins=int(binwidth), facet_col=facet_by, title=title, **aesthetics)
        fig.show()
    else:
        aesthetics.setdefault("color", "blue")
        fig, ax = plt.subplots()
        
        if facet_by:
            data.groupby(facet_by)[variable].apply(lambda x: x.hist(bins=int(binwidth), ax=ax, **aesthetics))
        else:
            data[variable].hist(bins=int(binwidth), ax=ax, **aesthetics)
        
        if title:
            ax.set_title(title)
        
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
create_histogram(data, 'variable1', interactive=True, facet_by='category', title="Interactive Histogram")
