import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

def quadrant_norm(data, x, y, r_label=True, save_path=None, interactive=False, annotate_quadrants=False, color='darkblue', fontstyle='italic'):
    """
    This function plots a scatter plot with several customizations like annotation of quadrants,
    Pearson's correlation coefficient and an option to save the plot.
    
    Args:
        data (pd.DataFrame): The data frame containing the data.
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.
        r_label (bool): Whether to display the Pearson's correlation coefficient. Defaults to True.
        save_path (str): The path to save the plot. If None, the plot is not saved. Defaults to None.
        interactive (bool): Whether to display the plot interactively. Defaults to False.
        annotate_quadrants (bool): Whether to annotate the quadrants with percentages. Defaults to False.
        color (str): The color for annotations. Defaults to 'darkblue'.
        fontstyle (str): The font style for annotations. Defaults to 'italic'.
    """
    
    if not isinstance(data, pd.DataFrame) or x not in data.columns or y not in data.columns:
        raise ValueError("Invalid data or column names")
    
    x_data, y_data = data[x], data[y]
    x_rescaled = (x_data - x_data.mean()) / x_data.std()
    y_rescaled = (y_data - y_data.mean()) / y_data.std()
    
    df_rescaled = pd.DataFrame({x_rescaled.name: x_rescaled, y_rescaled.name: y_rescaled})
    
    p = sns.lmplot(x=x_rescaled.name, y=y_rescaled.name, data=df_rescaled, scatter=True, ci=None, lowess=True)
    
    if r_label:
        r = round(pearsonr(x_data, y_data)[0], 2)
        plt.text(x_rescaled.min(), y_rescaled.max(), f'r = {r}', horizontalalignment='left', verticalalignment='top', fontstyle=fontstyle, color=color, alpha=0.6)
    
    if annotate_quadrants and len(x_rescaled) >= 4:
        quadrant_count = pd.crosstab(pd.cut(x_rescaled, bins=[-float('inf'), 0, float('inf')], include_lowest=True),
                                     pd.cut(y_rescaled, bins=[-float('inf'), 0, float('inf')], include_lowest=True))
        
        quadrant_pct = (quadrant_count / quadrant_count.sum().sum()) * 100
        quadrant_labels = [f"Q{i+1}: {quadrant_pct.iloc[i//2, i%2]:.1f}%" for i in range(4)]
        
        x_annotate = x_rescaled.quantile([0.25, 0.75, 0.25, 0.75])
        y_annotate = y_rescaled.quantile([0.75, 0.75, 0.25, 0.25])
        for i, (x_a, y_a) in enumerate(zip(x_annotate, y_annotate)):
            plt.text(x_a, y_a, quadrant_labels[i], horizontalalignment='left', verticalalignment='top', 
                     fontstyle=fontstyle, color=color, alpha=0.6)
    
    plt.axhline(0, linewidth=1, color='black')
    plt.axvline(0, linewidth=1, color='black')
    plt.xlabel(f"Rescaled {x}")
    plt.ylabel(f"Rescaled {y}")
    
    if save_path:
        plt.savefig(save_path)
    
    if interactive:
        plt.show()
    
    return p

# Generating synthetic data
np.random.seed(0)
data_size = 100
data = pd.DataFrame({
    'mpg': np.random.normal(20, 5, data_size),
    'disp': np.random.normal(200, 50, data_size)
})

# Using the function with synthetic data
quadrant_norm(data, 'mpg', 'disp')
quadrant_norm(data, 'mpg', 'disp', annotate_quadrants=True)
quadrant_norm(data, 'mpg', 'disp', r_label=False)
quadrant_norm(data, 'mpg', 'disp', save_path="rescaled_plot.png")
quadrant_norm(data, 'mpg', 'disp', interactive=True)