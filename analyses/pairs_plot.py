import seaborn as sns
import matplotlib.pyplot as plt

def pairs_plot(dataset_url, variables=None, plot_kind='scatter', hue=None, palette='husl', 
                      markers=None, size=2.5, height=2.5, aspect=1, diag_kind='auto', 
                      corner=False, log_scale=False, save_as=None, return_grid=False):
    """
    Creates a pairs plot for the specified variables in a dataset.

    Args:
    dataset_url (str): URL or filepath to the dataset.
    variables (list): List of variable names to include in the plot. If None, all numeric variables are included.
    plot_kind (str): Kind of plot to produce. Options are 'scatter' or 'kde'.
    hue (str): Variable in data to map plot aspects to different colors.
    palette (str): Color palette to use. Check seaborn color palettes for more options.
    markers (list): List of markers to use, or a single marker to use for all points.
    size (float): Size of the markers.
    height (float): Height of each facet in inches.
    aspect (float): Aspect ratio of each facet.
    diag_kind (str): Kind of plot for the diagonal subplots (hist, kde, or None).
    corner (bool): If True, don’t add the corner (diagonal) subplots.
    log_scale (bool): If True, plot variables in log scale.
    save_as (str): If specified, the plot will be saved to this filepath.
    return_grid (bool): If True, return the PairGrid instance for further customization.

    Returns:
    PairGrid instance if return_grid is True, else None
    """
    try:
        # Load dataset
        data = sns.load_dataset(dataset_url)

        # Apply log scale transformation if specified
        if log_scale:
            data[variables] = data[variables].apply(lambda x: np.log1p(x))

        # Create pairs plot
        grid = sns.pairplot(data, vars=variables, kind=plot_kind, hue=hue, palette=palette, 
                            markers=markers, plot_kws={"s": size}, height=height, aspect=aspect, 
                            diag_kind=diag_kind, corner=corner)

        # Save plot to file if save_as is specified
        if save_as:
            plt.savefig(save_as)
        
        # Show the plot
        plt.show()

        # Return the PairGrid instance if return_grid is True
        if return_grid:
            return grid
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
pairs_plot("iris", plot_kind='scatter', hue='species', palette='husl', markers=['o', 's', 'D'], size=50, height=2.5, aspect=1, diag_kind='kde', corner=True, log_scale=False, save_as='pairplot.png', return_grid=True)
