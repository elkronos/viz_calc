import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
import numpy as np

def donut_charts(df, show_labels=True):
    """
    This function plots a series of donut charts from the columns of a pandas DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input data frame where each column represents a different category.
    show_labels (bool): Determines whether or not to show the labels on the donut charts. 
                        Default is True.

    Returns:
    None: The function displays the donut charts plot.
    """
    
    font_color = '#FFFFFF'
    background_color = '#333333'
    
    # Dynamically generate a list of dark colors
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(df.columns)))
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), facecolor=background_color)
    fig.delaxes(axes[2, 2])

    color_cycle = cycle(colors)

    for i, (idx, row) in enumerate(df.head(8).iterrows()):
        ax = axes[i // 3, i % 3]
        ax.set_facecolor(background_color)
        row = row[row.gt(row.sum() * .01)]
        ax.pie(row, 
               labels=row.values if show_labels else None, 
               startangle=30, 
               wedgeprops=dict(width=.5), # For donuts
               colors=[next(color_cycle) for _ in row], 
               textprops={'color':font_color})
        ax.set_title(idx, fontsize=16, color=font_color)

    # Create a shared legend
    legend = fig.legend(df.columns, 
               bbox_to_anchor=(1.05, 0.7), 
               loc='upper left',  
               ncol=1, 
               fancybox=True, 
               facecolor=background_color,
               edgecolor=font_color)

    # Set the color of each text element in the legend
    for text in legend.get_texts():
        text.set_color(font_color)
        
    fig.subplots_adjust(wspace=.2, hspace=.5) # Space between charts

    title = fig.suptitle('Categories Analysis', y=.95, fontsize=20, color=font_color)
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.show()

# Generating a synthetic dataset for testing
def generate_synthetic_data():
    data = {'Category_A': [100, 200, 300, 400, 500, 600, 700, 800],
            'Category_B': [80, 270, 300, 450, 520, 610, 750, 810],
            'Category_C': [90, 220, 350, 410, 510, 670, 730, 830],
            'Category_D': [110, 230, 310, 420, 530, 660, 710, 820]}

    index_labels = ['Row1', 'Row2', 'Row3', 'Row4', 'Row5', 'Row6', 'Row7', 'Row8']
    
    return pd.DataFrame(data, index=index_labels)

# Test the function with synthetic data
df = generate_synthetic_data()
donut_charts(df, show_labels=False)
