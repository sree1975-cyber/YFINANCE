import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go

# Function to fetch stock data
def get_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data.reset_index()

# Function to calculate technical indicators and profit/loss
def calculate_technical_indicators(data):
    # Calculate moving averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

    # Calculate Profit and Loss
    data['Profit_Loss'] = data['Close'] - data['Open']
    data['Profit_Loss_Percentage'] = (data['Profit_Loss'] / data['Open']) * 100
    
    # Calculate RSI
    data['RSI'] = compute_rsi(data['Close'])
    
    return data

# Function to calculate RSI
def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to plot data
def plot_data(data, symbol):
    if data.empty:
        st.error("No data available for the specified date range.")
        return

    # Ensure necessary columns are present
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_columns):
        st.error("Data is missing required columns.")
        return

    fig = go.Figure()

    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick',
        hovertemplate='Open: %{open}<br>Close: %{close}<br>High: %{high}<br>Low: %{low}<br>Profit/Loss: %{customdata[0]:.2f}<br>Profit/Loss %: %{customdata[1]:.2f}<br><extra></extra>',
        customdata=np.column_stack((data['Profit_Loss'], data['Profit_Loss_Percentage']))
    ))

    # Add moving averages
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['SMA_20'],
        mode='lines', name='SMA 20',
        hovertemplate='SMA 20: %{y:.2f}<br>Date: %{x}<br><extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['SMA_50'],
        mode='lines', name='SMA 50',
        hovertemplate='SMA 50: %{y:.2f}<br>Date: %{x}<br><extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['EMA_20'],
        mode='lines', name='EMA 20',
        hovertemplate='EMA 20: %{y:.2f}<br>Date: %{x}<br><extra></extra>'
    ))

    # Configure layout
    fig.update_layout(
        title=f'Stock Data for {symbol}',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    return fig

def main():
    st.title('Stock Data Analysis')
    symbols_input = st.text_input('Enter stock symbols (comma-separated)', 'GOOGL,MSFT')
    symbols = [symbol.strip() for symbol in symbols_input.split(',')]
    start_date = st.date_input('Start Date', value=pd.to_datetime('2024-01-01'))
    end_date = st.date_input('End Date', value=pd.to_datetime('today'))

    if st.button('Fetch Data'):
        for symbol in symbols:
            data = get_stock_data(symbol, start_date, end_date)
            if not data.empty:
                data = calculate_technical_indicators(data)
                fig = plot_data(data, symbol)
                if fig:
                    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
