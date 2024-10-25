import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# Fetch historical stock data
def fetch_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data['Close']
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.Series()

# Detect sideways pattern
def detect_sideways_pattern(prices, threshold=0.02):
    pct_change = prices.pct_change()
    sideways = np.abs(pct_change) < threshold
    return sideways

# Calculate moving averages
def calculate_moving_averages(prices, short_window=20, long_window=50):
    short_mavg = prices.rolling(window=short_window).mean()
    long_mavg = prices.rolling(window=long_window).mean()
    return short_mavg, long_mavg

# Calculate Bollinger Bands
def calculate_bollinger_bands(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

# Magic stock and price findings
def magic_findings(prices):
    max_price = prices.max()
    min_price = prices.min()
    avg_price = prices.mean()
    return max_price, min_price, avg_price

# Plotting function with Plotly
def plot_data(prices, sideways, short_mavg, long_mavg, rolling_mean, upper_band, lower_band):
    fig = go.Figure()
    
    # Price trace
    fig.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines', name='Price', line=dict(color='blue')))
    
    # Sideways pattern trace
    fig.add_trace(go.Scatter(x=prices.index[sideways], y=prices[sideways], mode='markers', 
                             name='Sideways Pattern', marker=dict(color='orange', size=10)))
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=short_mavg.index, y=short_mavg, mode='lines', name='Short MA', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=long_mavg.index, y=long_mavg, mode='lines', name='Long MA', line=dict(color='red')))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, mode='lines', name='Rolling Mean', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=upper_band.index, y=upper_band, mode='lines', name='Upper Band', fill=None, 
                             line=dict(color='lightgray', dash='dash'), showlegend=True))
    fig.add_trace(go.Scatter(x=lower_band.index, y=lower_band, mode='lines', name='Lower Band', fill='tonexty', 
                             line=dict(color='lightgray', dash='dash'), showlegend=True))
    
    # Update layout
    fig.update_layout(title='Stock Price with Moving Averages and Bollinger Bands',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      hovermode='x unified')
    
    st.plotly_chart(fig)

# Streamlit UI
st.title('Advanced Market Analysis with Interactive Visualizations')

ticker = st.text_input('Enter stock ticker (e.g., AAPL):', 'AAPL')
start_date = st.date_input('Start date', pd.to_datetime('2020-01-01'))
end_date = st.date_input('End date', pd.to_datetime('today'))

# Moving average parameters
short_window = st.slider('Short Moving Average Window', min_value=5, max_value=100, value=20)
long_window = st.slider('Long Moving Average Window', min_value=5, max_value=100, value=50)

# Bollinger Bands parameters
bollinger_window = st.slider('Bollinger Bands Window', min_value=5, max_value=100, value=20)
num_std = st.slider('Bollinger Bands Std Dev', min_value=1, max_value=5, value=2)

if st.button('Analyze'):
    with st.spinner('Fetching data...'):
        prices = fetch_data(ticker, start_date, end_date)
        
        if prices.empty:
            st.error("No data found for this ticker.")
        else:
            sideways = detect_sideways_pattern(prices)
            short_mavg, long_mavg = calculate_moving_averages(prices, short_window, long_window)
            rolling_mean, upper_band, lower_band = calculate_bollinger_bands(prices, bollinger_window, num_std)
            max_price, min_price, avg_price = magic_findings(prices)

            st.subheader(f'{ticker} Analysis')
            plot_data(prices, sideways, short_mavg, long_mavg, rolling_mean, upper_band, lower_band)

            st.write(f'Total Days in Sideways Pattern: {sideways.sum()}')
            st.write(f'Magic Stock Findings: Max Price: {max_price:.2f}, Min Price: {min_price:.2f}, Average Price: {avg_price:.2f}')
