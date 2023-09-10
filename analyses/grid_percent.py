import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def grid_pecent(data, column_name, label=False, colors=("navy", "lightgray"), facet_var=None, facet_order=None):
    if column_name not in data.columns:
        raise ValueError("The specified column name is not present in the data frame")
    
    if facet_var is not None and facet_var not in data.columns:
        raise ValueError("The specified facet variable is not present in the data frame")

    if facet_var is None:
        facet_var = "facet_var"
        data[facet_var] = "All Data"

    if not isinstance(colors, (list, tuple)) or len(colors) != 2:
        raise ValueError("Colors argument should be a tuple or list with exactly two elements")

    facet_levels = data[facet_var].unique()

    if facet_order:
        if not set(facet_order).issubset(facet_levels):
            raise ValueError("The specified facet order contains levels not found in the facet variable")
        facet_levels = facet_order
    else:
        facet_levels = sorted(facet_levels)
    
    list_of_grids = []

    for facet_level in facet_levels:
        subset_data = data[data[facet_var] == facet_level]
        
        if subset_data[column_name].nunique() > 2:
            raise ValueError("Column data should be binary (0 or 1) for percentage calculation")

        perc = round(subset_data[column_name].sum() / len(subset_data) * 100, 2)
        
        grid_data = pd.DataFrame({'x': list(range(1, 11)) * 10, 'y': sorted(list(range(1, 11)) * 10)})
        filled_dots = min(100, max(0, round(perc)))
        grid_data['filled'] = [1 if i < filled_dots else 0 for i in range(100)]
        grid_data['facet_var'] = facet_level
        grid_data['label'] = f"{facet_level} - {column_name}: {perc}%"
        list_of_grids.append(grid_data)

    grid_data = pd.concat(list_of_grids)
    grid_data['facet_var'] = pd.Categorical(grid_data['facet_var'], categories=facet_levels)

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(len(facet_levels), 1, figure=fig)

    for i, facet_level in enumerate(facet_levels):
        ax = fig.add_subplot(gs[i, 0])
        subset_grid = grid_data[grid_data['facet_var'] == facet_level]
        colors_map = subset_grid['filled'].map({1: colors[0], 0: colors[1]})
        ax.scatter(subset_grid['x'], subset_grid['y'], c=colors_map, s=100, edgecolors='white', linewidths=0.5)
        ax.set_facecolor('floralwhite')
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(0.5, 10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(color='lightgray', linestyle='-', linewidth=0.5)

        if label:
            ax.set_title(subset_grid['label'].values[0], fontsize=16, fontweight='bold', fontfamily='serif', pad=12)

    plt.tight_layout()
    plt.show()

# Testing the function with varied distributions
data = pd.DataFrame({
    "percent_male": [0]*33 + [1]*67 + [0]*47 + [1]*53 + [0]*89 + [1]*11 + [0]*72 + [1]*28,
    "region": ["North"] * 100 + ["East"] * 100 + ["West"] * 100 + ["South"] * 100
})

grid_pecent(data, "percent_male", label=True, facet_var="region", facet_order=["North", "South", "East", "West"])
