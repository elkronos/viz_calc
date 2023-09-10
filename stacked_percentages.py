import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List

def stacked_percentages(
    data: pd.DataFrame, 
    group_var: str, 
    value_var: str, 
    color_palette: Union[str, List[str]] = 'viridis', 
    label_fontsize: int = 8, 
    label_color: str = 'white', 
    plot_title: str = "Percentages of one factor within another", 
    edgecolor: str = 'black', 
    figsize: tuple = (10, 6)
) -> pd.DataFrame:
    # Compute counts and percentages by group and value
    counts = data.groupby([group_var, value_var]).size().reset_index(name='n')
    total_counts = counts.groupby(group_var)['n'].transform('sum')
    counts['pct'] = (counts['n'] / total_counts) * 100

    # Create a bar plot of percentages
    plot_data = counts.pivot(index=group_var, columns=value_var, values='pct').reset_index()
    plot_data.set_index(group_var, inplace=True)
    plot_data.fillna(0, inplace=True)

    sns.set_theme(style="whitegrid")
    ax = plot_data.plot(kind='bar', stacked=True, colormap=color_palette, width=0.6, edgecolor=edgecolor, figsize=figsize)

    # Add percentage labels
    for i, patch in enumerate(ax.patches):
        if patch.get_height() >= 2:
            label_x_pos = patch.get_x() + patch.get_width() / 2
            label_y_pos = patch.get_y() + patch.get_height() / 2
            ax.text(label_x_pos, label_y_pos, f'{patch.get_height():.1f}%', ha='center', va='center', fontsize=label_fontsize, color=label_color, weight='bold')

    # Set plot properties
    plt.title(plot_title, fontsize=14, fontweight='bold')
    plt.xlabel(group_var, fontsize=12)
    plt.ylabel("Percentage", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.legend(title=value_var, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    return counts

def main():
    # Create a sample dataset
    data = {'color': ['D', 'E', 'F', 'G', 'D', 'E', 'F', 'G', 'D', 'E'],
            'cut': ['Ideal', 'Premium', 'Good', 'Fair', 'Ideal', 'Premium', 'Good', 'Fair', 'Fair', 'Fair']}

    df = pd.DataFrame(data)

    # Use the function and get the result
    result = stacked_percentages(df, 'color', 'cut', color_palette='Pastel1', label_fontsize=10, label_color='black')
    print(result)

if __name__ == "__main__":
    main()
