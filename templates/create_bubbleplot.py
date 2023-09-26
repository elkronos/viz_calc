import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_bubbleplot(df, x_col, y_col, size_col, label_col=None, color_col=None, xlabel=None, ylabel=None, title='Bubble Plot', save_filename=None, show_size_legend=False, base_size=500, size_scale=1000):
    """
    Create a bubble plot from a DataFrame.
    
    :param df: DataFrame containing the data
    :param x_col: Column name for x-axis values
    :param y_col: Column name for y-axis values
    :param size_col: Column name for bubble sizes
    :param label_col: Optional, column name for bubble labels
    :param color_col: Optional, column name for bubble colors
    :param xlabel: Optional, label for the x-axis
    :param ylabel: Optional, label for the y-axis
    :param title: Optional, title of the plot
    :param save_filename: Optional, filename to save the plot as an image file
    :param show_size_legend: Optional, whether to show a size legend
    :param base_size: Optional, base size for bubbles
    :param size_scale: Optional, scale factor for bubble sizes
    """
    x = df[x_col]
    y = df[y_col]
    normalized_size = (df[size_col] - df[size_col].min()) / (df[size_col].max() - df[size_col].min())
    size = normalized_size * size_scale + base_size
    
    labels = df[label_col] if label_col else None
    colors = df[color_col] if color_col else None
    
    xlabel = xlabel or x_col
    ylabel = ylabel or y_col
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(x, y, s=size, c=colors, alpha=0.6, edgecolors='w', linewidth=0.5)
    
    if labels is not None:
        for i, label in enumerate(labels):
            plt.annotate(label, (x.iloc[i], y.iloc[i]), textcoords="offset points", xytext=(0,0), ha='center', fontsize=12, color='black')
    
    if labels is not None:
        plt.legend(*scatter.legend_elements(), title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if show_size_legend:
        legend_sizes = [min(size), (max(size) + min(size)) / 2, max(size)]
        labels = [f'{(s-base_size)/size_scale:.2f}' for s in legend_sizes]
        size_legends = [plt.scatter([], [], s=s, label=label, color='gray', alpha=0.6) for s, label in zip(legend_sizes, labels)]
        plt.legend(handles=size_legends, title=size_col, bbox_to_anchor=(1.05, 0), loc='lower left')
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight')
    
    plt.show()

# Example usage:
np.random.seed(0)
n = 10
data = {
    'x': np.random.rand(n) * 10,
    'y': np.random.rand(n) * 10,
    'size': np.random.rand(n) * 100 + 10,
    'labels': [chr(65 + i) for i in range(n)],
    'colors': sns.color_palette("husl", n)
}
df = pd.DataFrame(data)

create_bubbleplot(df, 'x', 'y', 'size', 'labels', 'colors', show_size_legend=True, base_size=500, size_scale=1000)