import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def generate_synthetic_data(days=100, seed=42, start_price=100, noise_std_dev=1):
    """Generate a synthetic stock data set."""
    np.random.seed(seed)
    date_series = pd.date_range(start='1/1/2023', periods=days, freq='D')
    stock_prices = start_price + np.cumsum(np.random.normal(0, noise_std_dev, days))
    return pd.DataFrame({'Date': date_series, 'Stock_Price': stock_prices})

def create_stockplot(data, box_plot_interval=10, title='Synthetic Stock Data with Trends', save_as_html=False):
    """Plot the synthetic stock data with line plot and box plots for positive and negative trends."""
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1, 
        row_heights=[0.7, 0.3],
        subplot_titles=('Stock Prices', 'Box Plots Showing Trends'), 
        specs=[[{'type': 'scatter'}], [{'type': 'box'}]]
    )
    
    fig.add_trace(
        go.Scatter(x=data['Date'], y=data['Stock_Price'], mode='lines+markers', name='Stock Price', line=dict(color='blue')), 
        row=1, col=1
    )

    data['Trend'] = data['Stock_Price'].pct_change().apply(lambda x: 'Positive' if x > 0 else 'Negative')

    for i in range(0, len(data), box_plot_interval):
        interval = data[i:i+box_plot_interval]
        positive_trend = interval[interval['Trend'] == 'Positive']['Stock_Price']
        negative_trend = interval[interval['Trend'] == 'Negative']['Stock_Price']
        
        if not positive_trend.empty:
            fig.add_trace(go.Box(y=positive_trend, name=f'Pos {interval.Date.dt.date.iloc[0]}-{interval.Date.dt.date.iloc[-1]}', marker_color='green'), row=2, col=1)
        if not negative_trend.empty:
            fig.add_trace(go.Box(y=negative_trend, name=f'Neg {interval.Date.dt.date.iloc[0]}-{interval.Date.dt.date.iloc[-1]}', marker_color='red'), row=2, col=1)

    fig.update_layout(title=title, showlegend=True, template="plotly_dark")
    
    fig.show()

    if save_as_html:
        fig.write_html(f'{title}.html')

# Generate synthetic data and plot it
data = generate_synthetic_data(days=100, seed=42, start_price=100, noise_std_dev=1)
create_stockplot(data, box_plot_interval=10, title='Synthetic Stock Data with Trends', save_as_html=True)
