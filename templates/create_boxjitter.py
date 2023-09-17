import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def create_boxjitter(df, cont_var, cat_var, jitter_size=4, palette="coolwarm"):
    """
    This function plots a boxplot of a continuous variable across different categories of a categorical variable.
    It also overlays a jittered scatter plot on the boxplot to show the density of points.
    
    Parameters:
    df (pd.DataFrame): The data frame containing the data.
    cont_var (str): The column name of the continuous variable.
    cat_var (str): The column name of the categorical variable.
    jitter_size (float): The size of the points in the jittered scatter plot. Default is 4.
    palette (str): The palette to use for the boxplot. Default is "coolwarm".
    
    Returns:
    None
    """
    
    if cont_var not in df.columns or cat_var not in df.columns:
        raise ValueError(f"Columns {cont_var} or {cat_var} not found in the DataFrame")

    plt.figure(figsize=(12,8))
    
    sns.set_context("talk")
    sns.set_style("whitegrid", {'axes.grid': False})
    
    # Draw the boxplot
    sns.boxplot(x=cat_var, y=cont_var, data=df, palette=sns.color_palette("coolwarm", n_colors=df[cat_var].nunique()), width=0.5, boxprops=dict(alpha=.7))

    # Add jittered scatter plot
    sns.stripplot(x=cat_var, y=cont_var, data=df, jitter=True, size=jitter_size, hue=cat_var, palette=sns.color_palette("husl", n_colors=df[cat_var].nunique()), alpha=0.7)

    plt.title(f'Box plot of {cont_var} across different {cat_var}', fontsize=18, fontweight='bold')
    plt.xlabel(cat_var, fontsize=14, fontweight='bold')
    plt.ylabel(cont_var, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(labels=['Box Plot', 'Jitter Points'], loc='upper left', fontsize=12, title='Legend', title_fontsize='13')
    plt.show()

# Generate a synthetic dataset
np.random.seed(42)
data = {
    'continuous_var': np.random.randn(100),
    'categorical_var': np.random.choice(['A', 'B', 'C'], 100)
}
df = pd.DataFrame(data)

# Plot using the function
create_boxjitter(df, 'continuous_var', 'categorical_var')
