import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Fetch historical stock data
def fetch_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Detect sideways pattern
def detect_sideways_pattern(prices, threshold=0.02):
    pct_change = prices.pct_change()
    sideways = np.abs(pct_change) < threshold
    return sideways.astype(int)  # Return 1 for sideways, 0 otherwise

# Prepare features for machine learning
def prepare_features(data):
    features = pd.DataFrame()
    features['Close'] = data['Close']
    features['Pct_Change'] = features['Close'].pct_change()
    features['MA_10'] = features['Close'].rolling(window=10).mean()
    features['MA_20'] = features['Close'].rolling(window=20).mean()
    features['Volatility'] = features['Close'].rolling(window=5).std()
    features['Sideways'] = detect_sideways_pattern(data['Close'])
    features.dropna(inplace=True)
    return features

# Train machine learning model
def train_model(features):
    X = features[['Pct_Change', 'MA_10', 'MA_20', 'Volatility']]
    y = features['Sideways']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    
    return model, report

# Magic stock findings
def magic_findings(data):
    price_changes = data['Close'].pct_change() * 100
    significant_events = {
        "up_10_percent": data[price_changes >= 10],
        "down_10_percent": data[price_changes <= -10],
        "up_5_percent": data[price_changes >= 5],
        "down_5_percent": data[price_changes <= -5],
    }
    return significant_events

# Plotting function
def plot_data(data, model):
    # Prepare features for prediction
    features = prepare_features(data)
    
    # Ensure we have enough data for predictions
    if features.shape[0] > 0:
        data['Prediction'] = model.predict(features[['Pct_Change', 'MA_10', 'MA_20', 'Volatility']])
    
        fig = go.Figure()
        
        # Price trace
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price', line=dict(color='blue')))
        
        # Predictions trace
        fig.add_trace(go.Scatter(x=data.index, y=data['Prediction'] * 100, mode='lines', name='Predicted Sideways Pattern', line=dict(color='orange', dash='dash')))
        
        # Update layout
        fig.update_layout(title='Stock Price with Machine Learning Predictions',
                          xaxis_title='Date',
                          yaxis_title='Price',
                          hovermode='x unified')
        
        st.plotly_chart(fig)
    else:
        st.error("Not enough data to make predictions.")

# Streamlit UI
st.title('Market Analysis with Machine Learning for Sideways Patterns')

ticker = st.text_input('Enter stock ticker (e.g., AAPL):', 'AAPL')
start_date = st.date_input('Start date', pd.to_datetime('2020-01-01'))
end_date = st.date_input('End date', pd.to_datetime('today'))

if st.button('Analyze'):
    with st.spinner('Fetching data...'):
        data = fetch_data(ticker, start_date, end_date)
        
        if data.empty:
            st.error("No data found for this ticker.")
        else:
            features = prepare_features(data)
            model, report = train_model(features)
            st.write("### Model Classification Report")
            st.text(report)

            # Magic findings
            magic_events = magic_findings(data)
            st.write("### Magic Stock Findings")
            for event, details in magic_events.items():
                if not details.empty:
                    st.write(f"Significant Events - {event.replace('_', ' ').capitalize()}:")
                    st.table(details[['Open', 'High', 'Low', 'Close', 'Volume']])

            # Plot the data
            plot_data(data, model)

# Note: Make sure to run this code in a suitable Python environment with the required libraries.
