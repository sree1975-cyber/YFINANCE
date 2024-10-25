import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go

# Helper Functions
def get_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

def calculate_profit_loss(data):
    # Print the shapes and first few rows for debugging
    print("Data shape:", data.shape)
    print("Columns:", data.columns.tolist())
    print(data[['Open', 'Adj Close']].head())  # Check the relevant columns

    if 'Open' in data.columns and 'Adj Close' in data.columns:
        data['Profit-Loss'] = data['Adj Close'] - data['Open']
        data['Previous Close'] = data['Adj Close'].shift(1)

        # Fill NaN values for 'Previous Close'
        data['Previous Close'] = data['Previous Close'].fillna(method='bfill')

        # Debugging previous close
        print("Previous Close after fillna:", data['Previous Close'].head())

        # Calculate Adj/Open
        data['Adj/Open'] = data['Open'] - data['Previous Close']

        # Proceed with the rest of the calculations
        data['Gain_Loss'] = data['Profit-Loss'] + data['Adj/Open']

        conditions = [
            (data['Adj/Open'].isna()),
            (data['Adj/Open'] > 0),
            (data['Adj/Open'] < 0)
        ]
        choices = [
            (data['Gain_Loss'] / data['Open'] * 100).fillna(0),
            (data['Gain_Loss'] / (data['Open'] - data['Adj/Open'])) * 100,
            (data['Gain_Loss'] / (data['Open'] + data['Adj/Open'])) * 100
        ]
        data['%Change'] = np.select(conditions, choices, default=0)
        
    return data




def add_technical_indicators(data):
    # Moving Averages
    data['SMA_20'] = data['Adj Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Adj Close'].rolling(window=50).mean()
    
    # Relative Strength Index (RSI)
    delta = data['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['Upper Band'] = data['SMA_20'] + (data['Adj Close'].rolling(window=20).std() * 2)
    data['Lower Band'] = data['SMA_20'] - (data['Adj Close'].rolling(window=20).std() * 2)

    return data

def format_data(data):
    float_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Profit-Loss', 'Gain_Loss', 'Adj/Open', '%Change', 'SMA_20', 'SMA_50', 'RSI', 'Upper Band', 'Lower Band']
    data[float_columns] = data[float_columns].round(2)
    return data

def create_candlestick_chart(data, symbol):
    fig = go.Figure(data=[go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    )])

    # Add Moving Averages
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='blue')))
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Upper Band'], mode='lines', name='Upper Band', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Lower Band'], mode='lines', name='Lower Band', line=dict(color='green', dash='dash')))

    fig.update_layout(title=f'{symbol} Stock Price with Indicators', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
    
    return fig

def main():
    st.title('Stock Data Analysis with Technical Indicators')

    # Sidebar for Inputs
    with st.sidebar:
        symbols_input = st.text_input('Enter stock symbols (comma-separated)', 'GOOGL,MSFT')
        symbols = [symbol.strip() for symbol in symbols_input.split(',')]
        start_date = st.date_input('Start Date', value=pd.to_datetime('2024-01-01'))
        end_date = st.date_input('End Date', value=pd.to_datetime('today'))
        
        if st.button('Fetch Data'):
            st.session_state.stock_data = {}
            for symbol in symbols:
                data = get_stock_data(symbol, start_date, end_date)
                if not data.empty:
                    data = calculate_profit_loss(data)
                    data = add_technical_indicators(data)
                    st.session_state.stock_data[symbol] = data
                    st.write(f"Data fetched for {symbol}")
                else:
                    st.warning(f"No data found for {symbol}")

    # Main area for Data Display and Charts
    if 'stock_data' in st.session_state:
        for symbol, data in st.session_state.stock_data.items():
            formatted_data = format_data(data)
            st.write(f'{symbol} Stock Data')
            st.dataframe(formatted_data, use_container_width=True)

            fig = create_candlestick_chart(data, symbol)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
