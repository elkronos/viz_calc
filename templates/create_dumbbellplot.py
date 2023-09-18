import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def create_dumbbellplot(df, year_col, start_col, end_col, interactive=False, label=False, size=10, color_start='blue', color_end='red'):
    """
    Creates a dumbbell plot using either matplotlib or plotly.
    
    Parameters:
    df (pd.DataFrame): The data frame containing the data.
    year_col (str): The column name containing the year or categorical data.
    start_col (str): The column name for the starting values of the dumbbells.
    end_col (str): The column name for the ending values of the dumbbells.
    interactive (bool): If True, creates an interactive plot using plotly. If False, creates a plot using matplotlib. Default is False.
    label (bool): If True, labels the start and end points with their respective values. Default is False.
    size (int): The size of the scatter points. Default is 10.
    color_start (str): The color for the start points. Default is 'blue'.
    color_end (str): The color for the end points. Default is 'red'.
    """
    
    if interactive:
        # Prepare data for Plotly
        plot_data = []
        for _, row in df.iterrows():
            plot_data.append({year_col: row[year_col], 'Value': row[start_col], 'Label': row[start_col] if label else '', 'Color': color_start, 'Size': size})
            plot_data.append({year_col: row[year_col], 'Value': row[end_col], 'Label': row[end_col] if label else '', 'Color': color_end, 'Size': size})

        plot_df = pd.DataFrame(plot_data)

        # Create Plotly figure
        fig = px.scatter(plot_df, x='Value', y=year_col, color='Color', size='Size', text='Label')
        
        # Add lines connecting the points
        for _, row in df.iterrows():
            fig.add_shape(type='line', x0=row[start_col], y0=row[year_col], x1=row[end_col], y1=row[year_col], line=dict(color='gray'))
        
        # Adjust y-axis to display whole numbers only
        fig.update_yaxes(tickmode='array', tickvals=df[year_col].unique())

        # Show plot
        fig.show()
    else:
        # Create a plot using matplotlib
        plt.figure(figsize=(8,6))
        
        for _, row in df.iterrows():
            # Plot the points with larger size
            plt.scatter([row[start_col], row[end_col]], [row[year_col], row[year_col]], s=size*25, c=[color_start, color_end], zorder=2)
            
            # Add line connecting the points with thicker lines
            plt.plot([row[start_col], row[end_col]], [row[year_col], row[year_col]], color='gray', linewidth=2, zorder=1)
            
            # Add labels if the label parameter is set to True
            if label:
                plt.text(row[start_col], row[year_col], f"{row[start_col]}", va='center', ha='center', fontsize=8, fontweight='bold', color='white')
                plt.text(row[end_col], row[year_col], f"{row[end_col]}", va='center', ha='center', fontsize=8, fontweight='bold', color='white')

        # Set y-ticks to be integer values based on the year column data
        plt.yticks(df[year_col].unique(), fontsize=12)
        plt.xticks(fontsize=12)

        # Add titles and labels
        plt.title('Dumbbell Plot', fontsize=14, fontweight='bold')
        plt.xlabel('Value', fontsize=12)
        plt.ylabel(year_col, fontsize=12)

        # Show plot
        plt.tight_layout()
        plt.show()

# Example usage:
# Create a sample dataframe
data = {'Year': [2020, 2021, 2022], 'Start': [100, 150, 200], 'End': [200, 250, 300]}
df = pd.DataFrame(data)

# Call the function for non-interactive plot
create_dumbbellplot(df, 'Year', 'Start', 'End', interactive=False, label=True, size=20, color_start='green', color_end='red')

# Call the function for interactive plot
create_dumbbellplot(df, 'Year', 'Start', 'End', interactive=True, label=True, size=20, color_start='green', color_end='red')