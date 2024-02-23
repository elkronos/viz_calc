import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# Save function
def duration_plot(start_date, end_date, start_duration, end_duration, grouping_variable, data):
    data_sorted = data.sort_values(by=start_date)
    for col in [start_date, end_date, start_duration, end_duration]:
        data_sorted[col] = pd.to_datetime(data_sorted[col])

    date_range = data_sorted[end_date].max() - data_sorted[start_date].min()

    fig_height = max(6, len(data_sorted[grouping_variable]) * 0.4)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    overall_period = ax.barh(data_sorted[grouping_variable], (data_sorted[end_date] - data_sorted[start_date]), 
                             left=data_sorted[start_date], color='#6991AC', edgecolor='grey', label='Overall Period', alpha=0.7)
    duration = ax.barh(data_sorted[grouping_variable], (data_sorted[end_duration] - data_sorted[start_duration]), 
                       left=data_sorted[start_duration], color='#FFA07A', edgecolor='grey', label='Duration', alpha=0.9)

    if date_range <= pd.Timedelta(days=30):
        locator = mdates.DayLocator()
        formatter = mdates.DateFormatter('%d-%b')
    elif pd.Timedelta(days=30) < date_range <= pd.Timedelta(days=365):
        locator = mdates.MonthLocator()
        formatter = mdates.DateFormatter('%b-%Y')
    else:
        locator = mdates.YearLocator()
        formatter = mdates.DateFormatter('%Y')

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45)

    ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel(grouping_variable)
    ax.set_title('Group Duration by Date Range')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Example data
data = pd.DataFrame({
    'Group': ['A', 'B', 'C'],
    'Start_Date': ['2023-01-01', '2022-01-15', '2021-03-01'],
    'End_Date': ['2023-06-01', '2023-08-15', '2023-04-30'],
    'Start_Duration': ['2023-02-01', '2022-03-01', '2021-03-15'],
    'End_Duration': ['2023-05-01', '2023-07-01', '2023-04-01']
})

# Run example
duration_plot('Start_Date', 'End_Date', 'Start_Duration', 'End_Duration', 'Group', data)
