import pandas as pd
import matplotlib.pyplot as plt

# Helper function to abbreviate numbers
def abbreviate_number(num):
    if pd.isna(num):
        return "NA"
    elif num < 1000:
        return str(round(num, 1))
    elif num < 1e6:
        return f"{round(num/1e3, 1)}K"
    elif num < 1e9:
        return f"{round(num/1e6, 1)}M"
    elif num < 1e12:
        return f"{round(num/1e9, 1)}B"
    else:
        return f"{round(num/1e12, 1)}T"

# Function to set abbreviated labels on y-axis
def format_y_axis_labels(y, pos):
    return abbreviate_number(y)

# Function to create lollipop plot with abbreviated numbers
def lollipop_plot(data, x_var, y_var, statistic='mean', color='blue', labels=False, threshold=None, label_size=8, dot_size=800, line_width=1.2):
    if statistic == "mean":
        data = data.groupby(x_var, as_index=False).agg(aggregated=(y_var, 'mean'))
    elif statistic == "median":
        data = data.groupby(x_var, as_index=False).agg(aggregated=(y_var, 'median'))
    elif statistic == "sum":
        data = data.groupby(x_var, as_index=False).agg(aggregated=(y_var, 'sum'))
    else:
        data = data.groupby(x_var, as_index=False).agg(aggregated=(y_var, 'mean'))

    if labels:
        data['label'] = data['aggregated'].apply(abbreviate_number)

    if threshold is not None:
        data['alpha'] = data['aggregated'].apply(lambda x: 0.2 if x < threshold else 1)
        data['label_color'] = data['aggregated'].apply(lambda x: 'white' if x >= threshold else 'black')
    else:
        data['alpha'] = 1

    plt.figure()
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y_axis_labels))
    for idx, row in data.iterrows():
        plt.plot([row[x_var], row[x_var]], [0, row['aggregated']], color=color, linewidth=line_width, alpha=row['alpha'])
        plt.scatter(row[x_var], row['aggregated'], color=color, s=dot_size, alpha=row['alpha'])
        if labels:
            plt.text(row[x_var], row['aggregated'], row['label'], fontsize=label_size, color=row['label_color'], ha='center', va='bottom')

    if threshold is not None:
        plt.axhline(y=threshold, color='red', linestyle='--')

    plt.ylabel(statistic)
    plt.xlabel(x_var)
    plt.xticks(rotation=90)
    
    max_value = data['aggregated'].max()
    plt.ylim(0, max_value + max_value * 0.1)  # Set y-axis limits with a 10% padding at the top
    
    plt.tight_layout()
    plt.show()

# Generating synthetic data
import numpy as np
np.random.seed(123)
data = {
    'category': list('A'*1000 + 'B'*1000 + 'C'*1000 + 'D'*1000 + 'E'*1000),
    'values': np.concatenate([
        np.random.uniform(1, 40000, 1000),
        np.random.uniform(1, 40000, 1000),
        np.random.uniform(40001, 60000, 1000),
        np.random.uniform(60001, 100000, 1000),
        np.random.uniform(60001, 100000, 1000)
    ])
}
df = pd.DataFrame(data)

# Test with 'mean' statistic
lollipop_plot(df, 'category', 'values', labels=True, threshold=50000, statistic="mean")