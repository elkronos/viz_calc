import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import pearsonr, spearmanr

def _compute_matrix(data, method="pearson"):
    """
    Compute the correlation matrix for the given data.
    
    Parameters:
    - data (pd.DataFrame): Dataset for which correlation matrix is to be computed.
    - method (str, optional): Correlation method. Either 'pearson' or 'spearman'. Default is 'pearson'.
    
    Returns:
    - pd.DataFrame: Correlation matrix.
    """
    if method == "spearman":
        cor_matrix, _ = spearmanr(data, nan_policy="omit")
        return pd.DataFrame(cor_matrix, index=data.columns, columns=data.columns)
    return data.corr()

def _group_correlations(data, grouping_var, method="pearson"):
    """
    Group data by variable and compute their correlation matrices.
    
    Parameters:
    - data (pd.DataFrame): Dataset containing the data to be analyzed.
    - grouping_var (str): Name of the column used for grouping.
    - method (str, optional): Correlation method. Either 'pearson' or 'spearman'. Default is 'pearson'.
    
    Returns:
    - tuple: A tuple containing:
        - dict: Dictionary of correlation matrices, one for each group.
        - list: List of tuples. Each tuple contains two group names and their absolute difference matrix.
    """
    if grouping_var not in data.columns:
        raise ValueError(f"Error: Grouping variable {grouping_var} does not exist in the dataset.")
    
    cor_dict = {group: _compute_matrix(data.loc[data[grouping_var] == group].select_dtypes(include=[np.number]), method=method)
                for group in data[grouping_var].unique()}
    
    differences = [(g1, g2, abs(cor_dict[g1] - cor_dict[g2]))
                   for g1, g2 in combinations(cor_dict.keys(), 2)]
    
    return cor_dict, differences

def _visualize_difference_matrices(differences):
    """
    Display heatmaps of the differences between correlation matrices.
    
    Parameters:
    - differences (list): List of tuples. Each tuple contains two group names and their absolute difference matrix.
    """
    for g1, g2, diff in differences:
        mask = np.triu(np.ones_like(diff, dtype=bool))
        plt.figure(figsize=(8, 6))
        sns.heatmap(diff, annot=True, mask=mask, cmap="coolwarm", center=0, vmin=0, vmax=1)
        plt.title(f"Difference between {g1} and {g2}")
        plt.show()

def compare_correlations(data, grouping_var, method="pearson"):
    """
    Wrapper function to compute and visualize correlations within groups.
    
    Parameters:
    - data (pd.DataFrame): Input dataset.
    - grouping_var (str): Name of the column used for grouping.
    - method (str, optional): Correlation method. Either 'pearson' or 'spearman'. Default is 'pearson'.
    """
    differences = _group_correlations(data, grouping_var, method)[1]
    _visualize_difference_matrices(differences)

# Test with example data
np.random.seed(123)
x1 = np.random.randn(300)
x2 = np.random.randn(300)
x3 = x1 + np.random.randn(300) * 0.5
x4 = x2 + np.random.randn(300) * 0.5
grouping_var_values = np.random.choice(["Group1", "Group2", "Group3"], 300)
data = pd.DataFrame({"grouping_var": grouping_var_values, "x1": x1, "x2": x2, "x3": x3, "x4": x4})

compare_correlations(data, "grouping_var", method="spearman")
