import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def nested_pie(df, title="Nested Pie Chart", cmap_name="tab20c", figsize=(10, 7), edgecolor='w', linewidth=1, outer_labels=None, inner_labels=None, save_as=None):
    """
    Creates a nested pie chart from a pandas DataFrame.

    Parameters:
    - df (pd.DataFrame): Input data frame with numeric values. Each column represents a category and each row a subgroup.
    - title (str): Title of the plot. Default is 'Nested Pie Chart'.
    - cmap_name (str): The name of a colormap recognized by matplotlib. Default is 'tab20c'.
    - figsize (tuple): Figure size passed to `plt.subplots`. Default is (10, 7).
    - edgecolor (str): Edge color of the bars in the plot. Default is 'w' (white).
    - linewidth (int): Line width of the edges of the bars in the plot. Default is 1.
    - outer_labels (list): A list of labels for the outer categories. Default is None.
    - inner_labels (list): A list of labels for the inner categories. Default is None.
    - save_as (str): Filepath to save the plot as an image. Default is None, which means the plot will not be saved.

    Returns:
    - None: Displays the plot using plt.show().
    """
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")

    num_rows, num_cols = df.shape
    data = df.values
    norm = data / np.sum(data) * 2 * np.pi
    left = np.cumsum(np.append(0, norm.flatten()[:-1])).reshape(data.shape)

    cmap = plt.get_cmap(cmap_name)
    outer_colors = cmap(np.linspace(0, 1, num_rows))
    inner_colors = cmap(np.linspace(0, 1, num_rows * num_cols))

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    ax.bar(x=left[:, 0],
           width=norm.sum(axis=1),
           bottom=1-num_cols,
           height=num_cols,
           color=outer_colors,
           edgecolor=edgecolor,
           linewidth=linewidth,
           align="edge")
    
    # Adding labels for outer categories
    if outer_labels:
        for i, label in enumerate(outer_labels):
            ax.text(left[i, 0] + norm.sum(axis=1)[i]/2, 1-num_cols+num_cols/2, label, ha='center', va='center', color='white')
    
    ax.bar(x=left.flatten(),
           width=norm.flatten(),
           bottom=1-2*num_cols,
           height=num_cols,
           color=inner_colors,
           edgecolor=edgecolor,
           linewidth=linewidth,
           align="edge")
    
    # Adding labels for inner categories
    if inner_labels:
        for i, label in enumerate(inner_labels):
            ax.text(left.flatten()[i] + norm.flatten()[i]/2, 1-2*num_cols+num_cols/2, label, ha='center', va='center', color='black', size=6)

    ax.set(title=title)
    ax.set_axis_off()
    
    if save_as:
        plt.savefig(save_as)
    
    plt.show()

# Define a pandas DataFrame
df = pd.DataFrame({
    'A': [23, 17, 35, 29, 12, 41],
    'B': [16, 23, 11, 33, 27, 42],
})

# Define outer and inner labels
outer_labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5', 'Group 6']
inner_labels = [
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6'
]

# Call the function with labels
nested_pie(
    df, 
    outer_labels=outer_labels, 
    inner_labels=inner_labels,
    title="Nested Pie Chart with Labels"
)
