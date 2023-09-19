import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.stats import chi2

def plot_pca(data, group_column, feature_columns, components=[0, 1]):
    # Perform PCA
    pca = PCA(n_components=max(components)+1)
    pca_result = pca.fit_transform(data[feature_columns])
    
    # Add PCA result and group column to a new data frame
    result_df = pd.DataFrame(pca_result, columns=[f'PC{i}' for i in range(max(components)+1)])
    result_df['Group'] = data[group_column]
    
    # Set dark grid style and color palette
    sns.set_style("darkgrid")
    color_palette = sns.color_palette("Dark2", len(result_df['Group'].unique()))
    
    # Plot the PCA result with circles around each group
    plt.figure()
    ax = sns.scatterplot(x=f'PC{components[0]}', y=f'PC{components[1]}', hue='Group', data=result_df, palette=color_palette, edgecolor="w", s=100, style='Group', markers=True)
    ax.set_facecolor("#2e2e2e")
    plt.grid(color="grey", linestyle="--", linewidth=0.5)
    ax.set_aspect('equal', 'box')

    # Draw a circle around points in each group
    for idx, group in enumerate(result_df['Group'].unique()):
        group_data = result_df[result_df['Group'] == group]
        cov = np.cov(group_data[[f'PC{components[0]}', f'PC{components[1]}']].T)
        mean = group_data[[f'PC{components[0]}', f'PC{components[1]}']].mean(axis=0).values
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        
        # Calculate the angle and width and height of the ellipse based on the eigenvectors and eigenvalues
        angle = np.arctan2(v[1, 0], v[0, 0])
        width, height = 2 * np.sqrt(lambda_ * chi2.ppf(0.68, df=2))
        
        ell = Ellipse(xy=mean, width=width, height=height, angle=np.degrees(angle), edgecolor=color_palette[idx], linestyle='--')
        ell.set_facecolor('none')
        ax.add_artist(ell)
        
        # Add group annotation
        ax.text(mean[0], mean[1], f'{group}', fontsize=12, ha='center', va='center', color=color_palette[idx], weight='bold')
    
    plt.xlabel(f'Principal Component {components[0]}', color='white')
    plt.ylabel(f'Principal Component {components[1]}', color='white')
    plt.title('PCA Plot with Circles', color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# Load Iris dataset as an example
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Define feature columns and group column
feature_columns = iris.feature_names
group_column = 'target'

# Use the function to plot PCA with circles
plot_pca(iris_df, group_column, feature_columns)
