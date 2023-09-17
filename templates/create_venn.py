from typing import Iterable, List, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3

def create_venn(*args: Iterable[Union[int, str]], labels: List[str] = None, title: str = None, figsize: Tuple[int, int] = (8, 8), colors: List[str] = None, alpha: float = 0.4) -> None:
    """
    Create a Venn diagram with 1 to 3 sets.

    Args:
        *args: 1 to 3 iterables representing the sets.
        labels: List of labels for the sets, defaults to None.
        title: Title for the plot, defaults to None.
        figsize: Tuple representing the figure size, defaults to (8, 8).
        colors: List of colors for the different sections, defaults to None.
        alpha: Opacity of the colors, defaults to 0.4.

    Raises:
        ValueError: If the number of sets is not between 1 and 3 or the number of labels doesn't match the number of sets.
    """
    num_sets = len(args)
    if not 1 <= num_sets <= 3:
        raise ValueError("Provide between 1 and 3 iterables.")

    labels = labels or [f'Set {i+1}' for i in range(num_sets)]
    
    if len(labels) != num_sets:
        raise ValueError("The number of labels should match the number of sets.")

    sets = [set(arg) for arg in args]

    # Set up the plot with improved aesthetics
    plt.figure(figsize=figsize)
    plt.gca().set_facecolor('lightgray')
    plt.gca().add_patch(plt.Circle((0.5, 0.5), 0.5, color='white'))
    plt.grid(linestyle='--')

    # Choose the correct Venn function based on the number of sets
    venn_function = venn2 if num_sets < 3 else venn3
    venn = venn_function(sets, set_labels=labels, set_colors=colors, alpha=alpha)

    # Customize the plot title with better font style
    if title:
        plt.title(title, fontsize=16, fontweight='bold', color='darkblue')

    # Show the plot
    plt.show()

# Example usage:
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}
set3 = {7, 8, 9, 10}

create_venn(set1, set2, set3, labels=['A', 'B', 'C'], title='Sample Venn Diagram', colors=['red', 'green', 'blue'])
