import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from typing import Optional

def correlogram(data: pd.DataFrame, method: str = 'pearson', color: str = 'coolwarm', annotate: bool = False, significance: bool = False, p_threshold: float = 0.05, figsize: tuple = (10, 8), title: str = "Correlation Heatmap", save_as: Optional[str] = None, decimals: int = 2, triangle: str = 'full') -> None:
    """
    Computes a correlation heatmap with options for Pearson's or Spearman's correlations.

    :param data: A pandas DataFrame with the data.
    :param method: The correlation method ('pearson', 'spearman'). Default is 'pearson'.
    :param color: The color palette to use for the heatmap. Default is 'coolwarm'.
    :param annotate: Whether to annotate cells with the correlation values. Default is False.
    :param significance: Whether to annotate cells with significance (p-value). Default is False.
    :param p_threshold: The p-value threshold for significance. Default is 0.05.
    :param figsize: The size of the heatmap figure. Default is (10, 8).
    :param title: The title of the heatmap. Default is "Correlation Heatmap".
    :param save_as: The name of the file to save the heatmap as. If None, the plot will not be saved. Default is None.
    :param decimals: The number of decimal places to use for the correlation values in annotations. Default is 2.
    :param triangle: Which part of the triangle to display ('full', 'lower', 'upper'). Default is 'full'.

    :raises ValueError: If an invalid method or triangle parameter is provided.
    """
    if method not in ['pearson', 'spearman']:
        raise ValueError("Invalid method. Use 'pearson' or 'spearman'.")

    if triangle not in ['full', 'lower', 'upper']:
        raise ValueError("Invalid triangle parameter. Use 'full', 'lower', or 'upper'.")

    if method == 'pearson':
        corr_func = pearsonr
    else:
        corr_func = spearmanr

    corr = data.corr(method=method, numeric_only=True)
    p_vals = data.corr(method=lambda x, y: corr_func(x, y)[1], numeric_only=True)

    if annotate:
        annotations = corr.round(decimals).astype(str)
        if significance:
            significant = p_vals < p_threshold
            annotations[significant] = annotations[significant] + '*'
    else:
        annotations = None

    if triangle == 'lower':
        mask = np.triu(np.ones_like(corr, dtype=bool))
    elif triangle == 'upper':
        mask = np.tril(np.ones_like(corr, dtype=bool))
    else:
        mask = None

    sns.set(rc={'figure.figsize': figsize})
    ax = sns.heatmap(corr, annot=annotations, cmap=color, fmt='', vmin=-1, vmax=1, mask=mask)
    ax.set_title(title)

    if save_as:
        ax.get_figure().savefig(save_as)

# Generate a synthetic dataset
np.random.seed(0)
N = 100

base = np.linspace(0, 50, N)
feature1 = base + np.random.normal(0, 10, N) 
feature2 = 50 - base + np.random.normal(0, 10, N)
feature3 = np.sin(base/10) * 30 + np.random.normal(0, 10, N) 
feature4 = np.random.normal(0, 10, N) 

data = {
    'Feature1': feature1,
    'Feature2': feature2,
    'Feature3': feature3,
    'Feature4': feature4,
}
df = pd.DataFrame(data)

# Use the function to compute and plot the heatmap
correlogram(df, method='spearman', annotate=True, significance=True, triangle = 'lower')