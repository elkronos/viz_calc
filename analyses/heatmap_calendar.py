import numpy as np
import pandas as pd
import calmap
import matplotlib.pyplot as plt

def heatmap_calendar(date_values, start_date, cmap='coolwarm', fillcolor='lightgrey', linewidth=1, linecolor='white'):
    """
    Plots a series of heatmap calendars for each year within the given date range.

    Parameters:
    date_values (array-like): Array-like object containing values for each date.
    start_date (str): String representing the start date in 'YYYY-MM-DD' format.
    cmap (str, optional): Colormap for the heatmap. Defaults to 'coolwarm'.
    fillcolor (str, optional): Color for the days with no data. Defaults to 'lightgrey'.
    linewidth (float, optional): Line width of the grid. Defaults to 1.
    linecolor (str, optional): Line color of the grid. Defaults to 'white'.
    """
    try:
        # Convert date_values to a numpy array for efficiency
        date_values = np.asarray(date_values)
        
        # Getting the date range from the series
        all_days = pd.date_range(start=start_date, periods=len(date_values), freq='D')
        
        # Creating a series with the date range as the index
        events = pd.Series(date_values, index=all_days)
        
        # Finding the start and end years to determine how many calendars to plot
        start_year = all_days.year.min()
        end_year = all_days.year.max()
        
        # Looping through each year and plotting a calendar heatmap for that year
        for year in range(start_year, end_year+1):
            # Filtering events for the current year using loc
            year_events = events.loc[events.index.year == year]
            
            # Checking if there are any events for the current year, if yes, then plot
            if not year_events.empty:
                plt.figure(figsize=(16, 10))
                ax = calmap.yearplot(year_events, year=year, cmap=cmap, fillcolor=fillcolor, linewidth=linewidth, linecolor=linecolor)
                
                # Enhancing the fontsize and rotation of the labels
                ax.tick_params(axis='x', labelrotation=90, labelsize=10)
                ax.tick_params(axis='y', labelsize=10)
                
                plt.title(f'Heatmap Calendar for {year}', fontsize=16)
                plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage example
values = np.random.rand(700)  # Generating random values for 700 days
heatmap_calendar(values, '2022-01-01')
