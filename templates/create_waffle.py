import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List

def create_waffle(dataframe: pd.DataFrame, value_col: str, category_col: str, width: int = 20, height: int = 10, value_sign: str = '') -> None:
    """
    Creates and displays a waffle chart.

    Parameters:
    dataframe (pd.DataFrame): The dataframe containing the data to plot.
    value_col (str): The name of the column in the dataframe representing the values.
    category_col (str): The name of the column in the dataframe representing the categories.
    width (int): The width of the waffle chart. Default is 20.
    height (int): The height of the waffle chart. Default is 10.
    value_sign (str): A string to append to the values in the legend. Default is an empty string.

    Returns:
    None: The function displays the waffle chart using matplotlib and returns None.
    """

    total_values = sum(dataframe[value_col])
    total_num_tiles = width * height
    tiles_per_category = [round((value / total_values) * total_num_tiles) for value in dataframe[value_col]]

    # Correcting the rounding error
    while sum(tiles_per_category) < total_num_tiles:
        tiles_per_category[tiles_per_category.index(min(tiles_per_category))] += 1

    waffle_chart = np.zeros((height, width))

    category_index = 0
    tile_index = 0
    for category_tiles in tiles_per_category:
        for i in range(category_tiles):
            row, col = divmod(tile_index, width)
            waffle_chart[row][col] = category_index
            tile_index += 1
        category_index += 1

    colormap = plt.cm.viridis(np.linspace(0, 1, len(dataframe[category_col])))
    plt.matshow(waffle_chart, cmap=plt.cm.colors.ListedColormap(colormap))

    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, (height), 1), minor=True)

    plt.xticks([])
    plt.yticks([])

    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    legend_handles = [mpatches.Patch(color=colormap[i], label=f"{category} ({tiles_per_category[i]}{value_sign})") 
                      for i, category in enumerate(dataframe[category_col])]

    plt.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=len(dataframe[category_col]),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )

    plt.show()

# Generate a fake data set
np.random.seed(0)
data = {
    'group': ['Group A', 'Group B', 'Group C', 'Group D'],
    'nb_people': np.random.randint(1, 10, size=4)
}
df = pd.DataFrame(data)

# Use the waffle_plot function to generate a waffle plot
create_waffle(df, 'nb_people', 'group')
