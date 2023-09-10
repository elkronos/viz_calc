import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def create_dumbbellplot(df, year_col, start_col, end_col, interactive=False, label=False, size=10, color_start='blue', color_end='red'):
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
        plt.figure()
        
        for _, row in df.iterrows():
            # Plot the points
            plt.scatter([row[start_col], row[end_col]], [row[year_col], row[year_col]], s=size, c=[color_start, color_end])
            
            # Add line connecting the points
            plt.plot([row[start_col], row[end_col]], [row[year_col], row[year_col]], color='gray')
            
            # Add labels if the label parameter is set to True
            if label:
                plt.text(row[start_col], row[year_col], f"{row[start_col]}", va='center', ha='right')
                plt.text(row[end_col], row[year_col], f"{row[end_col]}", va='center', ha='left')

        # Set y-ticks to be integer values based on the year column data
        plt.yticks(df[year_col].unique())

        # Show plot
        plt.show()

# Example usage:
# Create a sample dataframe
data = {'Year': [2020, 2021, 2022], 'Start': [100, 150, 200], 'End': [200, 250, 300]}
df = pd.DataFrame(data)

# Call the function
create_dumbbellplot(df, 'Year', 'Start', 'End', interactive=True, label=True, size=20, color_start='green', color_end='red')
