import streamlit as st
import yfinance as yf
import pandas as pd

# Function to fetch stock data
def fetch_stock_data(symbols, start_date, end_date):
    data = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            # Fetch historical data within the specified date range
            data[symbol] = stock.history(start=start_date, end=end_date)
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
    return data

# Streamlit app
st.title("Stock Data Fetcher")

# Input for stock symbols
symbols_input = st.text_input("Enter stock symbols separated by commas (e.g., AAPL, MSFT, GOOG):")
symbols = [symbol.strip() for symbol in symbols_input.split(",") if symbol.strip()]

# Date inputs
start_date = st.date_input("Select start date:")
end_date = st.date_input("Select end date:")

# Button to fetch data
if st.button("Fetch Stock Data"):
    if symbols and start_date and end_date:
        stock_data = fetch_stock_data(symbols, start_date, end_date)
        
        # Display data for each stock
        for symbol, data in stock_data.items():
            st.subheader(f"Data for {symbol} from {start_date} to {end_date}")
            if not data.empty:
                st.dataframe(data)
            else:
                st.warning(f"No data available for {symbol} in the specified date range.")
    else:
        st.warning("Please enter at least one stock symbol and select the date range.")

