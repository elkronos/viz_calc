import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def bench_bar(data, x_var, y_var, use_two_colors=True, threshold="mean", 
              y_limits=None, colors=("yellow", "blue"), error_type="se"):
    """
    This function takes a pandas DataFrame and plots a bar chart with error bars.
    
    Parameters:
    data (pd.DataFrame): Input data frame
    x_var (str): Column name to be used on x-axis
    y_var (str): Column name to be used on y-axis
    use_two_colors (bool): Whether to use two colors for bars based on the threshold
    threshold (str/int/float): Value or method for calculating the threshold
    y_limits (tuple): Y-axis limits
    colors (tuple): Tuple with two color values
    error_type (str): Error type ('se' for standard error or 'sd' for standard deviation)
    
    Returns:
    None
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data should be a pandas DataFrame")

    if x_var not in data.columns or y_var not in data.columns:
        raise ValueError(f"{x_var} or {y_var} not found in the data columns")

    if not isinstance(colors, tuple) or len(colors) != 2:
        raise ValueError("colors should be a tuple with exactly two elements")

    assert data[y_var].dtype in ["int64", "float64"], f"{y_var} is not numeric"

    means = data.groupby(x_var).agg(
        mean=pd.NamedAgg(column=y_var, aggfunc='mean'),
        se=pd.NamedAgg(column=y_var, aggfunc=lambda x: np.std(x, ddof=1)/np.sqrt(len(x))),
        sd=pd.NamedAgg(column=y_var, aggfunc='std')
    ).reset_index()

    if threshold == "mean":
        threshold_value = data[y_var].mean()
    elif threshold == "median":
        threshold_value = data[y_var].median()
    elif isinstance(threshold, (int, float)):
        threshold_value = threshold
    else:
        raise ValueError("Invalid threshold value")

    if error_type == "se":
        error_col = "se"
        error_text = "standard errors"
    elif error_type == "sd":
        error_col = "sd"
        error_text = "standard deviations"
    else:
        raise ValueError("Invalid error type")

    fig, ax = plt.subplots()

    if use_two_colors:
        bar_colors = [colors[1] if x >= threshold_value else colors[0] for x in means['mean']]
    else:
        bar_colors = 'blue'

    ax.bar(means[x_var], means['mean'], yerr=means[error_col], color=bar_colors, capsize=5)
    
    ax.axhline(y=threshold_value, color='r', linestyle='--', label=f'Threshold = {threshold_value:.2f}')

    ax.set_title(f'Average of {y_var} by each level of {x_var}')
    ax.set_xlabel(x_var, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_var, fontsize=14, fontweight='bold')

    if y_limits:
        ax.set_ylim(y_limits)
    
    plt.subplots_adjust(bottom=0.25)
    ax.text(1.05, -0.25, f'Threshold = {threshold_value:.2f}, Error bars represent {error_text}', 
            verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, 
            fontsize=10, fontstyle='italic')

    plt.tight_layout()
    plt.show()

# Example usage
data = pd.DataFrame({
    'group': ['A']*20 + ['B']*20 + ['C']*20,
    'value': np.concatenate([np.random.normal(10, 2, 20), np.random.normal(12, 2, 20), np.random.normal(15, 2, 20)])
})

bench_bar(data, 'group', 'value')