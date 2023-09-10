import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_timeseries_fill(data, time_col, series_cols, colors=None, fill_between=False, 
                         title=None, xlab=None, ylab=None, alpha_val=0.5, line_width=2):
    
    if fill_between and len(series_cols) != 2:
        raise ValueError("fill_between option requires exactly two series.")
    
    if colors is None:
        colors = ['blue', 'red']
    
    # Define which line is on top for coloring the bars
    data['top_color'] = data.apply(lambda row: colors[0] if row[series_cols[0]] >= row[series_cols[1]] else colors[1], axis=1)
    
    # Initialize the plot
    plt.figure()
    
    # Add lines for each series
    for i, series_col in enumerate(series_cols):
        plt.plot(data[time_col], data[series_col], color=colors[i], linewidth=line_width)
    
    # Add bars between the lines if fill_between is TRUE
    if fill_between:
        for i in range(len(data)):
            plt.plot([data[time_col].iloc[i], data[time_col].iloc[i]], 
                     [data[series_cols[0]].iloc[i], data[series_cols[1]].iloc[i]], 
                     color=data['top_color'].iloc[i], 
                     alpha=alpha_val, 
                     linewidth=line_width/2)
    
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True)
    plt.show()

# Synthetic data example with less rigid crossovers
time = pd.date_range(start="2021-01-01", periods=100, freq='D')
series1 = 10 + np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(scale=0.5, size=100)
series2 = 10 + np.cos(np.linspace(0, 4*np.pi, 100)) + np.random.normal(scale=0.5, size=100)
data = {"time": time, "series1": series1, "series2": series2}
df = pd.DataFrame(data)

# Use the function with synthetic data
plot_timeseries_fill(df, "time", ["series1", "series2"], fill_between=True, colors=["blue", "red"], title="Crossover Example", xlab="Date", ylab="Value")
