import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

def significant_means(group_var=None, numeric_var=None, alpha=0.05, colors=['blue', 'orange'], axis_label_size=12, value_labels=True):
    """
    This function plots the mean values of two groups along with their standard error of the mean (SEM). 
    It also performs a t-test to check if the difference between the means of the two groups is significant.

    Parameters:
    -----------
    group_var : pandas.Series, optional
        A pandas series containing the group variable which separates the data into two groups (group 0 and group 1).
        It cannot be None, and it is expected to contain values 0 and 1.
    
    numeric_var : pandas.Series, optional
        A pandas series containing the numeric data variable. This variable is used to compute the means and SEM.
        It cannot be None.
    
    alpha : float, default=0.05
        The significance level used in the t-test to determine if the means are significantly different. 
        If the p-value obtained from the t-test is less than alpha, a significant difference is assumed.
    
    colors : list, default=['blue', 'orange']
        A list containing two strings representing the colors to be used for the two bar plots corresponding to group 0 and group 1.

    axis_label_size : int, default=12
        The fontsize to be used for the axis labels and annotations.

    value_labels : bool, default=True
        A flag to indicate whether the mean values should be displayed at the top of the bars.

    Returns:
    --------
    None

    Raises:
    -------
    ValueError
        If either group_var or numeric_var is None.
    """
    if group_var is None or numeric_var is None:
        raise ValueError("group_var and numeric_var cannot be None")

    group1_data = numeric_var[group_var == 0].dropna()
    group2_data = numeric_var[group_var == 1].dropna()

    means = [group1_data.mean(), group2_data.mean()]
    errors = [group1_data.sem(), group2_data.sem()]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(['Group 1', 'Group 2'], means, yerr=errors, color=colors, capsize=10, alpha=0.75)
    
    y_max = max(means) + max(errors) + 0.5
    
    if value_labels:
        ax.text(0, means[0] + errors[0] + 0.1, f'{means[0]:.2f}', ha='center', va='bottom', fontsize=axis_label_size)
        ax.text(1, means[1] + errors[1] + 0.1, f'{means[1]:.2f}', ha='center', va='bottom', fontsize=axis_label_size)
    
    t_stat, p_val = stats.ttest_ind(group1_data, group2_data)
    if p_val < alpha:
        ax.annotate("", xy=(0, y_max), xycoords='data',
                    xytext=(1, y_max), textcoords='data',
                    arrowprops=dict(arrowstyle="-", ec='black',
                                    connectionstyle="bar,fraction=0.2"))
        ax.text(0.5, y_max + 0.2, f'* (p={p_val:.3f})', ha='center', va='bottom', fontsize=axis_label_size)
        y_max += 0.4  # Adjusting y_max to make space for the bracket and label

    ax.set_ylabel('Mean Value', fontsize=axis_label_size)
    ax.set_xlabel('Group', fontsize=axis_label_size)
    ax.set_title('Mean Value by Group', fontsize=axis_label_size+2, y=1.02 + (0.02 * y_max))  # Adjusted the y-position dynamically based on y_max
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

# Synthetic data to test the function
data = {
    'group_var': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    'significant_difference': [6.1, 3.3, 2.9, 4.0, 4.9, 5.5, 3.2, 6.0, 12.9, 19.5, 11.2, 14.8, 12.1, 15.3, 15.7, 11.0],
    'no_significant_difference': [7.1, 7.3, 7.0, 7.2, 7.5, 7.1, 7.3, 7.2, 7.1, 7.3, 7.4, 7.2, 7.3, 7.0, 7.1, 7.2]
}

df = pd.DataFrame(data)
significant_means(group_var=df['group_var'], numeric_var=df['significant_difference'])
significant_means(group_var=df['group_var'], numeric_var=df['no_significant_difference'])
