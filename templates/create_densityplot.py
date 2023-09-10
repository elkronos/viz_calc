import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to create density plot
def create_densityplot(data, x=None, hue=None, col=None, row=None, kde_kws=None, 
                        palette=None, facet_grid=False, x_label=None, y_label=None, title=None):
    sns.set(style="whitegrid")
    
    if facet_grid:
        g = sns.FacetGrid(data, col=col, row=row, hue=hue, palette=palette, margin_titles=True)
        g.map(sns.kdeplot, x, **(kde_kws or {})).add_legend()
        if title:
            g.fig.suptitle(title, y=1.02)
    else:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=data, x=x, hue=hue, shade=True, palette=palette, **(kde_kws or {}))
        if x_label: plt.xlabel(x_label)
        if y_label: plt.ylabel(y_label)
        if title: plt.title(title)
        if hue: plt.legend()
        plt.show()

# Generate synthetic data
np.random.seed(42)
data_size = 1000
data_dict = {
    'Value': np.concatenate([np.random.normal(5, 1, data_size), np.random.normal(10, 2, data_size)]),
    'Group': ['A'] * data_size + ['B'] * data_size,
    'Subgroup': np.concatenate([np.random.choice(['X', 'Y'], data_size), np.random.choice(['X', 'Y'], data_size)])
}
data = pd.DataFrame(data_dict)

# Print the first few rows of the data
print(data.head())

# Use the synthetic data to create density plots
create_densityplot(data, x="Value", hue="Group", x_label="Value", y_label="Density", title="Density Plot Example")

# With facet grid
create_densityplot(data, x="Value", hue="Group", col="Subgroup", facet_grid=True, title="Faceted Density Plot Example")
