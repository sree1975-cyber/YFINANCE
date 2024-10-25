import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

def get_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data.reset_index()

def calculate_profit_loss(data):
    required_columns = ['Open', 'Adj Close']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {set(required_columns) - set(data.columns)}")
    
    data['Profit-Loss'] = data['Adj Close'] - data['Open']
    data['Previous Close'] = data['Adj Close'].shift(1).fillna(method='bfill')
    data['Adj/Open'] = data['Open'] - data['Previous Close'].fillna(0)
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

def main():
    st.title('Stock Data Analysis')
    symbols_input = st.text_input('Enter stock symbols (comma-separated)', 'GOOGL,MSFT')
    symbols = [symbol.strip() for symbol in symbols_input.split(',')]
    start_date = st.date_input('Start Date', value=pd.to_datetime('2024-01-01'))
    end_date = st.date_input('End Date', value=pd.to_datetime('today'))

    if st.button('Fetch Data'):
        for symbol in symbols:
            data = get_stock_data(symbol, start_date, end_date)
            data = calculate_profit_loss(data)
            st.write(data)

if __name__ == "__main__":
    main()
