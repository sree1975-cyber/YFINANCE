# 2_EVENT_STUDY_WITH_PERCENT_CHANGE.py( THIS PROGRAM IS BASED ON CLOSE PRICE, BUT i MODIFIED THE BELOW PROGRAM TO USE ADJ CLOSE PRICE FOR PROFIT-LOSS)

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
    # Calculate Profit-Loss
    data['Profit-Loss'] = data['Adj Close'] - data['Open']
    
    # Calculate Previous Close from the Adj Close
    data['Previous Close'] = data['Adj Close'].shift(1)  # Get the previous day's adjusted close
    
    # Calculate Adj/Open
    data['Adj/Open'] = data['Open'] - data['Previous Close']

    # Calculate Gain_Loss
    data['Gain_Loss'] = data['Profit-Loss'] + data['Adj/Open'].fillna(0)

    # Calculate %Change
    conditions = [
        (data['Adj/Open'].isna()),  # If Adj/Open is None
        (data['Adj/Open'] > 0),      # If Adj/Open is positive
        (data['Adj/Open'] < 0)       # If Adj/Open is negative
    ]
    
    choices = [
        (data['Gain_Loss'] / data['Open'] * 100).fillna(0),  # For Condition 1: use Profit-Loss for %Change
        (data['Gain_Loss'] / (data['Open'] - data['Adj/Open'])) * 100,  # For Condition 2
        (data['Gain_Loss'] / (data['Open'] + data['Adj/Open'])) * 100   # For Condition 3
    ]
    
    data['%Change'] = np.select(conditions, choices, default=0)  # Default to 0 if none match
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
        return ''  # Default to no color for non-numeric values
    
    # Apply color coding to the specified columns
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
    
    # Highlighting highest and lowest closing prices
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

    # Highlighting highest and lowest high/low prices
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

    # Annotating Open Price > Previous Close Price
    data['Previous Close'] = data['Adj Close'].shift(1)
    greater_open_price_records = data[data['Open'] > data['Previous Close']]
    fig.add_trace(go.Scatter(
        x=greater_open_price_records['Date'],
        y=greater_open_price_records['Open'],
        mode='markers',
        marker=dict(color='purple', size=8),
        name='Open > Previous Close'
    ))

    # Annotating Open Price < Previous Close Price
    less_open_price_records = data[data['Open'] < data['Previous Close']]
    fig.add_trace(go.Scatter(
        x=less_open_price_records['Date'],
        y=less_open_price_records['Open'],
        mode='markers',
        marker=dict(color='cyan', size=8),
        name='Open < Previous Close'
    ))

    # Customizing the layout
    fig.update_layout(
        title=f'{symbol} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',  # Unified hover mode to show data for all traces
        legend=dict(x=0, y=1.15, orientation='h'),  # Legend position
        autosize=True
    )
    
    return fig

def main():
    st.title('Stock Data Analysis')
    # Custom CSS for wider output
    st.markdown("""
        <style>
        .dataframe-container {
            width: 100% !important;
            overflow-x: auto; /* Allow horizontal scroll if needed */
        }
        .plotly-graph-div {
            width: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar for Inputs and Analysis Options
    with st.sidebar:
        st.header('Input Parameters')
        symbols_input = st.text_input('Enter stock symbols (comma-separated)', 'GOOGL,MSFT')
        symbols = [symbol.strip() for symbol in symbols_input.split(',')]
        start_date = st.date_input('Start Date', value=pd.to_datetime('2024-01-01'))
        end_date = st.date_input('End Date', value=pd.to_datetime('today'))

        if st.button('Fetch Data'):
            st.session_state.stock_data = {}
            for symbol in symbols:
                st.session_state.stock_data[symbol] = get_stock_data(symbol, start_date, end_date)
                st.session_state.stock_data[symbol] = calculate_profit_loss(st.session_state.stock_data[symbol])
                st.write(f"Data fetched for {symbol}")

            st.session_state.data_fetched = True
            
        st.subheader('Analysis Options')
        with st.expander('Filter Options', expanded=True):
            select_all = st.checkbox('Select All Analysis Options', key='select_all')

            # Individual analysis checkboxes
            highest_closing_checkbox = st.checkbox('Find highest closing price value record', value=st.session_state.get('highest_closing', False))
            lowest_closing_checkbox = st.checkbox('Find lowest closing price value record', value=st.session_state.get('lowest_closing', False))
            highest_profit_checkbox = st.checkbox('Find records with highest profit', value=st.session_state.get('highest_profit', False))
            lowest_profit_checkbox = st.checkbox('Find records with lowest profit', value=st.session_state.get('lowest_profit', False))  
            highest_loss_checkbox = st.checkbox('Find records with highest loss', value=st.session_state.get('highest_loss', False))
            lowest_loss_checkbox = st.checkbox('Find records with lowest loss', value=st.session_state.get('lowest_loss', False))   
            highest_high_checkbox = st.checkbox('Find highest value in the high column', value=st.session_state.get('highest_high', False))
            lowest_low_checkbox = st.checkbox('Find lowest value in the low column', value=st.session_state.get('lowest_low', False))   
            open_greater_checkbox = st.checkbox('Open price > Previous day adj close price', value=st.session_state.get('open_greater', False))
            open_less_checkbox = st.checkbox('Open price < Previous day adj close price', value=st.session_state.get('open_less', False))

            # Handle Select All
            if select_all:
                # Set all checkboxes to True if "Select All" is checked
                st.session_state.highest_closing = True
                st.session_state.lowest_closing = True
                st.session_state.highest_profit = True
                st.session_state.lowest_profit = True
                st.session_state.highest_loss = True
                st.session_state.lowest_loss = True
                st.session_state.highest_high = True
                st.session_state.lowest_low = True
                st.session_state.open_greater = True
                st.session_state.open_less = True
            else:
                # Set the session state based on individual checkboxes
                st.session_state.highest_closing = highest_closing_checkbox
                st.session_state.lowest_closing = lowest_closing_checkbox
                st.session_state.highest_profit = highest_profit_checkbox
                st.session_state.lowest_profit = lowest_profit_checkbox
                st.session_state.highest_loss = highest_loss_checkbox
                st.session_state.lowest_loss = lowest_loss_checkbox
                st.session_state.highest_high = highest_high_checkbox
                st.session_state.lowest_low = lowest_low_checkbox
                st.session_state.open_greater = open_greater_checkbox
                st.session_state.open_less = open_less_checkbox
                    
            apply_filters_button = st.button('Apply Filters')
            new_analysis_button = st.button('New Analysis')

        if new_analysis_button:
            st.session_state.stock_data = {}
            st.session_state.data_fetched = False
            st.session_state.clear()

    # Main area for Data Display and Charts
    if 'data_fetched' in st.session_state and st.session_state.data_fetched:
        st.header('Stock Data and Charts')

        # Layout for horizontal display with expanded width
        col1, col2 = st.columns([3.99, 0.01])  # Adjust proportions as needed
        
        with col1:
            for symbol in symbols:
                if symbol in st.session_state.stock_data:
                    formatted_data = format_data(st.session_state.stock_data[symbol])
                    st.write(f'{symbol} Stock Data')
                    # Use container width for dataframe display
                    st.dataframe(formatted_data, use_container_width=True)
                    
                    fig = create_candlestick_chart(st.session_state.stock_data[symbol], symbol)
                    st.plotly_chart(fig, use_container_width=True)

        # Analytical Results displayed at the bottom of the page
        st.header('Analytical Results')
        if apply_filters_button:
            for symbol in symbols:
                if symbol in st.session_state.stock_data:
                    data = st.session_state.stock_data[symbol]
                                               
                    # Analysis based on session state checkboxes
                    if st.session_state.get('highest_closing', False):
                        max_close_price = data['Adj Close'].max()
                        max_close_record = data[data['Adj Close'] == max_close_price]
                        st.subheader(f'{symbol} Highest Closing Price Record')
                        st.dataframe(max_close_record, use_container_width=True)
                        if not select_all:
                            break  # Stop if not "Select All"

                    if st.session_state.get('lowest_closing', False):
                        min_close_price = data['Adj Close'].min()
                        min_close_record = data[data['Adj Close'] == min_close_price]
                        st.subheader(f'{symbol} Lowest Closing Price Record')
                        st.dataframe(min_close_record, use_container_width=True)
                      
                    if st.session_state.get('highest_high', False):
                        max_high_price = data['High'].max()
                        max_high_record = data[data['High'] == max_high_price]
                        st.subheader(f'{symbol} Highest High Price Record')
                        st.dataframe(max_high_record, use_container_width=True)
                    
                    if st.session_state.get('lowest_low', False):
                        min_low_price = data['Low'].min()
                        min_low_price_record = data[data['Low'] == min_low_price]
                        st.subheader(f'{symbol} Lowest Low Price Record')
                        st.dataframe(min_low_price_record, use_container_width=True)
                   
                    if st.session_state.get('open_greater', False):
                        greater_open_price_records = data[data['Open'] > data['Previous Close']]
                        st.subheader(f'{symbol} Open Price >(Greater than) Previous Close Price Records')
                        st.dataframe(greater_open_price_records, use_container_width=True)
              
                    if st.session_state.get('open_less', False):
                        less_open_price_records = data[data['Open'] < data['Previous Close']]
                        st.subheader(f'{symbol} Open Price <(Less than) Previous Close Price Records')
                        st.dataframe(less_open_price_records, use_container_width=True)
                      
                    if st.session_state.get('highest_profit', False):
                        profits = data[data['Profit-Loss'] > 0]
                        if not profits.empty:  # Check if there are any profits
                            highest_profit = profits['Profit-Loss'].max()  # Get maximum profit
                            highest_profit_record = profits[profits['Profit-Loss'] == highest_profit]
                            st.subheader(f'{symbol} Highest Profit Record')
                            st.dataframe(highest_profit_record, use_container_width=True)
                        else:
                            st.subheader(f'{symbol} No Profit Records Found')  # Inform if there are no profits

                    if st.session_state.get('lowest_profit', False):
                        positive_profits = data[data['Profit-Loss'] > 0]
                        if not positive_profits.empty:  # Check if there are any positive profits
                            lowest_profit = positive_profits['Profit-Loss'].min()
                            lowest_profit_record = positive_profits[positive_profits['Profit-Loss'] == lowest_profit]
                            st.subheader(f'{symbol} Lowest Profit Record')
                            st.dataframe(lowest_profit_record, use_container_width=True)
                        else:
                            st.subheader(f'{symbol} No Positive Profit Records Found')  # Inform if there are no positive profits

                    if st.session_state.get('highest_loss', False):
                        losses = data[data['Profit-Loss'] < 0]
                        if not losses.empty:  # Check if there are any losses
                            highest_loss = losses['Profit-Loss'].min()  # Get the most negative loss
                            highest_loss_record = losses[losses['Profit-Loss'] == highest_loss]
                            st.subheader(f'{symbol} Highest Loss Record')
                            st.dataframe(highest_loss_record, use_container_width=True)
                        else:
                            st.subheader(f'{symbol} No Loss Records Found')  # Inform if there are no losses

                    if st.session_state.get('lowest_loss', False):
                        losses = data[data['Profit-Loss'] < 0]
                        if not losses.empty:  # Check if there are any losses
                            lowest_loss = losses['Profit-Loss'].max()  # Get the least negative loss
                            lowest_loss_record = losses[losses['Profit-Loss'] == lowest_loss]
                            st.subheader(f'{symbol} Lowest Loss Record')
                            st.dataframe(lowest_loss_record, use_container_width=True)
                        else:
                            st.subheader(f'{symbol} No Loss Records Found')  # Inform if there are no losses

if __name__ == "__main__":
    main()
