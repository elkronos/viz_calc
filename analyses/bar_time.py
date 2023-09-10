import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def bar_time(data, date_col, y_col, group_by='month', window_size=7, bar_stat='sum', line_stat='mean'):
    
    if not isinstance(date_col, str) or not isinstance(y_col, str) or not isinstance(group_by, str):
        raise ValueError("The date_col, y_col, and group_by arguments must be strings.")

    if not isinstance(window_size, int):
        raise ValueError("The window_size argument must be an integer.")

    if bar_stat not in ['sum', 'mean', 'median']:
        raise ValueError("The bar_stat argument must be one of 'sum', 'mean', or 'median'.")
    
    if line_stat not in ['mean', 'median', 'max', 'min']:
        raise ValueError("The line_stat argument must be one of 'mean', 'median', 'max', or 'min'.")

    data[date_col] = pd.to_datetime(data[date_col])
    
    freq = {'day': 'D', 'month': 'M', 'quarter': 'Q', 'year': 'Y'}
    data_grouped = data.groupby(pd.Grouper(key=date_col, freq=freq[group_by])).agg({y_col: [bar_stat, line_stat]})
    
    print("Data grouped structure:")
    print(data_grouped.head()) 

    data_grouped.columns = [f'{y_col}_{bar_stat}', f'{y_col}_{line_stat}_line']
    data_grouped.reset_index(inplace=True)

    print("Data grouped after resetting index:")
    print(data_grouped.head())

    data_grouped['moving_stat'] = data_grouped[f'{y_col}_{line_stat}_line'].rolling(window=window_size, min_periods=1).mean()
    
    bar_width = {'day': 1, 'month': 30, 'quarter': 90, 'year': 365}[group_by]
    
    fig, ax = plt.subplots()
    ax.bar(data_grouped[date_col], data_grouped[f'{y_col}_{bar_stat}'], width=bar_width, color='steelblue', alpha=0.8)
    ax.plot(data_grouped[date_col], data_grouped['moving_stat'], color='red', linewidth=1)

    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'Time Series Bar Plot with rolling statistics')
    plt.tight_layout()
    plt.show()

# Generating a synthetic data set
data = pd.DataFrame({
    'date': pd.date_range(start='2010-01-01', end='2020-12-31', freq='D'),
    'value': np.random.normal(10, 2, 4018)
})

# Example usage
bar_time(data, date_col='date', y_col='value', group_by='month', window_size=3, bar_stat='mean', line_stat='mean')
