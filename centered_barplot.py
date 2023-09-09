import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def centered_barplot(data, x, y, threshold=None, add_labels=True, add_title=False):
    
    # Step 1: Calculate the threshold if not provided
    if threshold is None:
        threshold = data[y].mean()
    
    # Step 2: Prepare data summary
    data_grouped = data.groupby(x)[y]
    data_summary = data_grouped.agg(
        above_threshold=lambda group: (group >= threshold).sum() / group.count(),
        below_threshold=lambda group: (group < threshold).sum() / group.count()
    ).reset_index()

    # Step 3: Create initial plot
    fig, ax = plt.subplots(figsize=(10,6))

    bars1 = ax.bar(data_summary[x], data_summary['above_threshold'], color="#1F78B4", label='Above')
    bars2 = ax.bar(data_summary[x], -data_summary['below_threshold'], color="#FF7F00", label='Below')

    # Set y-axis labels to percentages
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Draw horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='solid')

    # Set labels and title
    ax.set_xlabel(f'Levels of {x}')
    ax.set_ylabel('Percentage')
    
    if add_title:
        ax.set_title(f"Percentage of {y} above and below {threshold:.2f} for each level of {x}", loc='center', pad=20, fontsize=12, color="#3D3D3D")
    
    if add_labels:
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.0%}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = -bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, -height, f'{height:.0%}', ha='center', va='top', fontsize=8)

    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    import numpy as np
    np.random.seed(123)
    example_data = pd.DataFrame({
        'Channel': np.random.choice(list('ABCDE'), 100),
        'Sales': np.random.normal(50, 15, 100)
    })
    
    centered_barplot(example_data, "Channel", "Sales", add_title=True)
