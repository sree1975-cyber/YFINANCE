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

# Magic stock findings
def magic_findings(prices):
    price_changes = prices.pct_change() * 100
    significant_events = {
        "up_10_percent": prices[(price_changes >= 10)],
        "down_10_percent": prices[(price_changes <= -10)],
        "up_5_percent": prices[(price_changes >= 5)],
        "down_5_percent": prices[(price_changes <= -5)],
    }
    return significant_events

# Calculate performance analysis for percentage movements
def performance_analysis(prices):
    changes = prices.pct_change() * 100
    days_to_gain_5 = (changes[changes >= 5].index[0] - changes.index[0]).days if any(changes >= 5) else None
    days_to_gain_10 = (changes[changes >= 10].index[0] - changes.index[0]).days if any(changes >= 10) else None
    return days_to_gain_5, days_to_gain_10

# Calculate support and resistance levels
def support_resistance(prices):
    support = prices.min()
    resistance = prices.max()
    return support, resistance

# Plotting function with Plotly
def plot_data(prices, sideways, short_mavg, long_mavg, rolling_mean, upper_band, lower_band, support, resistance, magic_events):
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

    # Support and Resistance Lines
    fig.add_trace(go.Scatter(x=prices.index, y=[support]*len(prices), mode='lines', name='Support', 
                             line=dict(color='purple', dash='dash')))
    fig.add_trace(go.Scatter(x=prices.index, y=[resistance]*len(prices), mode='lines', name='Resistance', 
                             line=dict(color='gold', dash='dash')))
    
    # Update layout
    fig.update_layout(title='Stock Price with Moving Averages, Bollinger Bands, Support, and Resistance',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      hovermode='x unified')
    
    st.plotly_chart(fig)

# Streamlit UI
st.title('Advanced Market Analysis with Magic Price Events and Dow Theory')

ticker = st.text_input('Enter stock ticker (e.g., AAPL):', 'AAPL')
start_date = st.date_input('Start date', pd.to_datetime('2020-01-01'))
end_date = st.date_input('End date', pd.to_datetime('today'))

# Time period selection
time_period = st.selectbox('Select Time Period for Analysis', ['5 Days', '15 Days', '1 Month', '3 Months', '6 Months', 'YTD', '1 Year', '2 Years', '3 Years', '4 Years', '5 Years', 'Max', 'Custom'])

# Custom date selection
if time_period == 'Custom':
    custom_start = st.date_input('Custom Start Date', pd.to_datetime('2020-01-01'))
    custom_end = st.date_input('Custom End Date', pd.to_datetime('today'))
else:
    custom_start, custom_end = pd.to_datetime(start_date), pd.to_datetime(end_date)

# Moving average parameters
short_window = st.slider('Short Moving Average Window', min_value=5, max_value=100, value=20)
long_window = st.slider('Long Moving Average Window', min_value=5, max_value=100, value=50)

# Bollinger Bands parameters
bollinger_window = st.slider('Bollinger Bands Window', min_value=5, max_value=100, value=20)
num_std = st.slider('Bollinger Bands Std Dev', min_value=1, max_value=5, value=2)

if st.button('Analyze'):
    with st.spinner('Fetching data...'):
        prices = fetch_data(ticker, custom_start, custom_end)
        
        if prices.empty:
            st.error("No data found for this ticker.")
        else:
            sideways = detect_sideways_pattern(prices)
            short_mavg, long_mavg = calculate_moving_averages(prices, short_window, long_window)
            rolling_mean, upper_band, lower_band = calculate_bollinger_bands(prices, bollinger_window, num_std)
            magic_events = magic_findings(prices)
            days_to_gain_5, days_to_gain_10 = performance_analysis(prices)
            support, resistance = support_resistance(prices)

            st.subheader(f'{ticker} Analysis')
            plot_data(prices, sideways, short_mavg, long_mavg, rolling_mean, upper_band, lower_band, support, resistance, magic_events)

            st.write(f'Total Days in Sideways Pattern: {sideways.sum()}')
            st.write(f'Magic Stock Findings:')
            st.write(f'Upward Events (≥ 10%): {magic_events["up_10_percent"]}')
            st.write(f'Downward Events (≤ -10%): {magic_events["down_10_percent"]}')
            st.write(f'Upward Events (≥ 5%): {magic_events["up_5_percent"]}')
            st.write(f'Downward Events (≤ -5%): {magic_events["down_5_percent"]}')
            st.write(f'Days to Gain ≥ 5%: {days_to_gain_5}, Days to Gain ≥ 10%: {days_to_gain_10}')
            st.write(f'Support Level: {support:.2f}, Resistance Level: {resistance:.2f}')
