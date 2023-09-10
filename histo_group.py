import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Function definition
def histo_group(df, num_var, group_var=None, bin_width=None, add_density=False, title=None, x_label=None, y_label='Count', palette='Set1', stat_line=None, custom_stat_value=None, facet_var=None):
    if bin_width is None:
        bin_width = 'auto'
    else:
        bin_width = int((df[num_var].max() - df[num_var].min()) / bin_width)
    
    if x_label is None:
        x_label = num_var
    
    g = sns.FacetGrid(df, col=facet_var, hue=group_var, palette=palette, height=4, aspect=2)
    
    if add_density:
        g.map(sns.kdeplot, num_var, fill=True, alpha=0.5)
    else:
        g.map(plt.hist, num_var, bins=bin_width, alpha=0.5)
    
    if stat_line is not None:
        if stat_line == "mean":
            stat_value = df[num_var].mean()
        elif stat_line == "median":
            stat_value = df[num_var].median()
        elif custom_stat_value is not None:
            stat_value = custom_stat_value
        else:
            raise ValueError("Invalid value for stat_line. Accepted values are 'mean', 'median', or use custom_stat_value for a custom value.")
        
        for ax in g.axes.flat:
            ax.axvline(stat_value, color='black', linestyle='--')

    g.set_axis_labels(x_label, y_label)
    if title:
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(title)
    
    plt.show()

# Create a synthetic dataset
np.random.seed(123)
data = {
    'mpg': np.random.normal(50, 15, 100),
    'type': np.random.choice(['A', 'B'], 100),
    'gear': np.random.choice([3, 4, 5], 100)
}
df = pd.DataFrame(data)

# Usage example
histo_group(df, 'mpg', group_var='type', add_density=False, stat_line=None, facet_var= 'gear')
