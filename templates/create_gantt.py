import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
import seaborn as sns

# Set a modern plot style
sns.set_style("whitegrid")

# Enhanced Function to create a Gantt chart
def create_gantt(df, col_mappings, color_map='viridis', plot_params=None):
    """
    Creates a Gantt chart from a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data
    col_mappings (dict): Dictionary with mappings for required columns: {'task_col': 'Task', 'start_col': 'Start', 'end_col': 'End'}
    color_map (str, optional): Name of the matplotlib color map to use for coloring tasks. Default is 'viridis'.
    plot_params (dict, optional): Additional parameters for plot customization.

    Returns:
    None
    """
    
    for col in col_mappings.values():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' specified is not in the dataframe")
    
    # Calculate the duration of each task
    df['Duration'] = df[col_mappings['end_col']] - df[col_mappings['start_col']]

    # Create a list of unique tasks
    tasks = df[col_mappings['task_col']].unique()
    
    # Set up the color mapping
    cmap = plt.get_cmap(color_map, len(tasks))

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, len(tasks) * 0.6))

    # Loop through each task and plot it on the Gantt chart
    for i, task in enumerate(tasks):
        task_data = df[df[col_mappings['task_col']] == task]
        for _, row in task_data.iterrows():
            ax.barh(task, width=row['Duration'], left=row[col_mappings['start_col']], height=0.4, color=cmap(i), edgecolor='none')
            ax.plot([row[col_mappings['start_col']], row[col_mappings['end_col']]], [task, task], color='black', linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Task')
    ax.set_title('Gantt Chart')

    # Setting x-axis to show dates clearly
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Remove the y-axis legend to declutter
    ax.yaxis.set_visible(False)
    
    # Setting xlim to cover the entire data range and adding task labels beside the bars
    min_date = df[col_mappings['start_col']].min()
    max_date = df[col_mappings['end_col']].max()
    ax.set_xlim(min_date, max_date)
    for i, task in enumerate(tasks):
        ax.text(min_date, i, task, va='center', ha='right', color='black')
    
    # Show the Gantt chart with additional customization
    plt.tight_layout()
    plt.xticks(rotation=45)
    if plot_params:
        plt.setp(ax, **plot_params)
    plt.show()

# Generating a synthetic dataset
data = {
    'Task': ['Task {}'.format(i) for i in range(1, 11) for _ in range(3)],
    'Start': [(datetime(2023, 9, 20) + timedelta(days=random.randint(0, 10), hours=random.randint(0, 23))) for _ in range(30)],
    'End': [(datetime(2023, 9, 20) + timedelta(days=random.randint(11, 20), hours=random.randint(0, 23))) for _ in range(30)],
}

df = pd.DataFrame(data)

# Testing the function
create_gantt(df, {'task_col': 'Task', 'start_col': 'Start', 'end_col': 'End'})
