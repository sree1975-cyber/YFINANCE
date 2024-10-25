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
    data['Profit-Loss'] = data['Adj Close'] - data['Open']
    data['Previous Close'] = data['Adj Close'].shift(1)
    data['Adj/Open'] = data['Open'] - data['Previous Close']
    data['Gain_Loss'] = data['Profit-Loss'] + data['Adj/Open'].fillna(0)

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

def format_data(data):
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.date
    float_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Profit-Loss', 'Gain_Loss', 'Adj/Open', '%Change']
    data[float_columns] = data[float_columns].round(2)

    def color_text(val):
        if isinstance(val, (int, float)):
            return 'color: #006400; font-weight: bold' if val >= 0 else 'color: #FF0000; font-weight: bold'
        return ''

    return data.style.map(color_text, subset=['Profit-Loss', 'Adj/Open', 'Gain_Loss', '%Change'])

def create_candlestick_chart(data, symbol):
    fig = go.Figure(data=[go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    )])

    # Highlight highest and lowest closing prices
    max_close_price = data['Adj Close'].max()
    max_close_date = data[data['Adj Close'] == max_close_price]['Date'].values[0]
    fig.add_trace(go.Scatter(
        x=[max_close_date],
        y=[max_close_price],
        mode='markers+text',
        marker=dict(color='lime', size=10),
        text=['Highest Close'],
        textposition='top right',
        name='Highest Close'
    ))

    min_close_price = data['Adj Close'].min()
    min_close_date = data[data['Adj Close'] == min_close_price]['Date'].values[0]
    fig.add_trace(go.Scatter(
        x=[min_close_date],
        y=[min_close_price],
        mode='markers+text',
        marker=dict(color='red', size=10),
        text=['Lowest Close'],
        textposition='bottom right',
        name='Lowest Close'
    ))

    # Highlighting high and low prices
    max_high_price = data['High'].max()
    max_high_date = data[data['High'] == max_high_price]['Date'].values[0]
    fig.add_trace(go.Scatter(
        x=[max_high_date],
        y=[max_high_price],
        mode='markers+text',
        marker=dict(color='blue', size=10),
        text=['Highest High'],
        textposition='top center',
        name='Highest High'
    ))

    min_low_price = data['Low'].min()
    min_low_date = data[data['Low'] == min_low_price]['Date'].values[0]
    fig.add_trace(go.Scatter(
        x=[min_low_date],
        y=[min_low_price],
        mode='markers+text',
        marker=dict(color='orange', size=10),
        text=['Lowest Low'],
        textposition='bottom center',
        name='Lowest Low'
    ))

    data['Previous Close'] = data['Adj Close'].shift(1)
    greater_open_price_records = data[data['Open'] > data['Previous Close']]
    fig.add_trace(go.Scatter(
        x=greater_open_price_records['Date'],
        y=greater_open_price_records['Open'],
        mode='markers',
        marker=dict(color='purple', size=8),
        name='Open > Previous Close'
    ))

    less_open_price_records = data[data['Open'] < data['Previous Close']]
    fig.add_trace(go.Scatter(
        x=less_open_price_records['Date'],
        y=less_open_price_records['Open'],
        mode='markers',
        marker=dict(color='cyan', size=8),
        name='Open < Previous Close'
    ))

    fig.update_layout(
        title=f'{symbol} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(x=0, y=1.15, orientation='h'),
        autosize=True
    )

    return fig

def main():
    st.title('Stock Data Analysis')

    # Sidebar for Inputs
    with st.sidebar:
        st.header('Input Parameters')
        symbols_input = st.text_input('Enter stock symbols (comma-separated)', 'GOOGL,MSFT')
        symbols = [symbol.strip() for symbol in symbols_input.split(',')]
        start_date = st.date_input('Start Date', value=pd.to_datetime('2024-01-01'))
        end_date = st.date_input('End Date', value=pd.to_datetime('today'))

        if st.button('Fetch Data'):
            st.session_state.stock_data = {}
            for symbol in symbols:
                data = get_stock_data(symbol, start_date, end_date)
                st.session_state.stock_data[symbol] = calculate_profit_loss(data)
                st.write(f"Data fetched for {symbol}")

    # Main area for Data Display and Charts
    if 'stock_data' in st.session_state:
        st.header('Stock Data and Charts')
        col1, col2 = st.columns([3.99, 0.01])

        with col1:
            for symbol in symbols:
                if symbol in st.session_state.stock_data:
                    formatted_data = format_data(st.session_state.stock_data[symbol])
                    st.write(f'{symbol} Stock Data')
                    st.dataframe(formatted_data, use_container_width=True)

                    fig = create_candlestick_chart(st.session_state.stock_data[symbol], symbol)
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
