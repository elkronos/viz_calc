import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
from adjustText import adjust_text
import ipywidgets as widgets

def animated_bubble(df, time_col, x_col, y_col, size_col, color_col, label_col, interval='day', bubble_scale=5000):
    """
    Generates an animated bubble plot.

    Parameters:
    df (pd.DataFrame): The data frame containing the data.
    time_col (str): The column name representing time.
    x_col (str): The column name representing the x-axis values.
    y_col (str): The column name representing the y-axis values.
    size_col (str): The column name representing the sizes of the bubbles.
    color_col (str): The column name representing the colors of the bubbles.
    label_col (str): The column name representing the labels of the bubbles.
    interval (str): The time interval for animation (day, week, month, quarter, year). Default is 'day'.
    bubble_scale (int): The scaling factor for the bubble sizes. Default is 5000.

    Returns:
    HTML object: An HTML object containing the animated bubble plot.
    """
    
    time_intervals = {
        'day': 'D',
        'week': 'W',
        'month': 'M',
        'quarter': 'Q',
        'year': 'Y'
    }
    
    unique_dates = pd.date_range(start=df[time_col].min(), end=df[time_col].max(), freq=time_intervals[interval])
    unique_colors = df[color_col].unique()
    color_map = {color: idx for idx, color in enumerate(unique_colors)}
    
    fig, ax = plt.subplots()
    
    progress = widgets.FloatProgress(value=0, min=0, max=len(unique_dates), description='Loading:', bar_style='info')
    display(progress)
    
    def update(frame):
        ax.clear()
        current_time = unique_dates[frame]
        mask = df[time_col] == current_time
        current_data = df[mask]
        
        texts = []
        for color_value in unique_colors:
            color_data = current_data[current_data[color_col] == color_value]
            sizes = bubble_scale * (color_data[size_col] / color_data[size_col].max())  
            scatter = ax.scatter(color_data[x_col], color_data[y_col], s=sizes, 
                                 color=plt.cm.viridis(color_map[color_value] / len(unique_colors)), label=color_value, 
                                 alpha=0.6, edgecolors='w')

            for i in range(len(color_data)):
                texts.append(ax.text(color_data[x_col].iloc[i], color_data[y_col].iloc[i], color_data[label_col].iloc[i], fontsize=9, ha='right'))
        
        adjust_text(texts)
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel(y_col.capitalize())
        ax.set_title(f'Time: {current_time.strftime("%Y-%m-%d")}', fontsize=12)
        ax.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        progress.value += 1

    ani = FuncAnimation(fig, update, frames=len(unique_dates), repeat=False)
    
    plt.close(fig)
    return HTML(ani.to_jshtml())

# Example usage
dates = pd.date_range(start='1/1/2020', periods=100, freq='D')
companies = [f'Company {i}' for i in range(15)]
industries = ['Industry1', 'Industry2', 'Industry3']

data = {
    'time': [],
    'Marketing Spend': [],
    'Sales': [],
    'Marketing Efficiency': [],
    'Industry': [],
    'Company': []
}

for date in dates:
    for company in companies:
        industry = np.random.choice(industries)
        marketing_spend = np.random.rand() * 1000
        sales = np.random.rand() * 10000
        efficiency = marketing_spend / sales
        data['time'].append(date)
        data['Marketing Spend'].append(marketing_spend)
        data['Sales'].append(sales)
        data['Marketing Efficiency'].append(efficiency)
        data['Industry'].append(industry)
        data['Company'].append(company)

df = pd.DataFrame(data)
animated_bubble(df, time_col='time', x_col='Marketing Spend', y_col='Sales', size_col='Marketing Efficiency', color_col='Industry', label_col='Company', interval='day')
