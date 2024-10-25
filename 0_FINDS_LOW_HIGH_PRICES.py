import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch historical stock data
def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data['Close']

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

# Plotting function
def plot_data(prices, sideways, short_mavg, long_mavg, rolling_mean, upper_band, lower_band):
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(prices.index, prices, label='Price', color='blue')
    plt.scatter(prices.index[sideways], prices[sideways], color='orange', label='Sideways Pattern', s=10)
    plt.plot(short_mavg.index, short_mavg, label='Short MA', color='green')
    plt.plot(long_mavg.index, long_mavg, label='Long MA', color='red')
    plt.title('Stock Price with Sideways Pattern and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(prices.index, prices, label='Price', color='blue')
    plt.plot(rolling_mean.index, rolling_mean, label='Rolling Mean', color='orange')
    plt.fill_between(upper_band.index, upper_band, lower_band, color='lightgray', alpha=0.5, label='Bollinger Bands')
    plt.title('Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()

    st.pyplot(plt)

# Streamlit UI
st.title('Market Analysis with Advanced Pattern Detection')

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
            
            st.subheader(f'{ticker} Sideways Pattern and Moving Averages Analysis')
            plot_data(prices, sideways, short_mavg, long_mavg, rolling_mean, upper_band, lower_band)
            st.write(f'Total Days in Sideways Pattern: {sideways.sum()}')
