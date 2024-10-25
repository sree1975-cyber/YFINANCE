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

# Calculate additional technical indicators
def calculate_technical_indicators(data):
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = compute_rsi(data['Close'])
    data['MACD'], data['MACD_Signal'] = compute_macd(data['Close'])
    return data

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, macd_signal

# Prepare features for machine learning
def prepare_features(data):
    features = pd.DataFrame()
    features['Close'] = data['Close']
    features['Pct_Change'] = features['Close'].pct_change()
    features['Volatility'] = features['Close'].rolling(window=5).std()
    features['Sideways'] = detect_sideways_pattern(data['Close'])
    features.dropna(inplace=True)
    return features

# Detect sideways pattern
def detect_sideways_pattern(prices, threshold=0.02):
    pct_change = prices.pct_change()
    sideways = np.abs(pct_change) < threshold
    return sideways.astype(int)  # Return 1 for sideways, 0 otherwise

# Train machine learning model
def train_model(features):
    X = features[['Pct_Change', 'Volatility']]
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

# Generate advisory based on predictions and indicators
def generate_advisory(data, model):
    latest_data = data.iloc[-1]
    prediction = model.predict([[latest_data['Pct_Change'], latest_data['Volatility']]])[0]

    # Basic advisory logic
    advisory = ""
    if prediction == 1:
        advisory += "BUY: The stock shows potential for a sideways trend.\n"
    else:
        advisory += "HOLD: No significant movements expected.\n"

    # Incorporate technical indicators
    if latest_data['RSI'] < 30:
        advisory += "Consider buying; RSI indicates oversold conditions.\n"
    elif latest_data['RSI'] > 70:
        advisory += "Consider selling; RSI indicates overbought conditions.\n"
    
    if latest_data['MACD'] > latest_data['MACD_Signal']:
        advisory += "MACD indicates upward momentum.\n"
    elif latest_data['MACD'] < latest_data['MACD_Signal']:
        advisory += "MACD indicates downward momentum.\n"
    
    # Dow Theory analysis
    advisory += dow_theory_analysis(data)
    
    return advisory

# Dow Theory analysis
def dow_theory_analysis(data):
    trends = []
    if data['Close'].iloc[-1] > data['MA_50'].iloc[-1]:
        trends.append("The stock is in an uptrend according to Dow Theory.")
    elif data['Close'].iloc[-1] < data['MA_50'].iloc[-1]:
        trends.append("The stock is in a downtrend according to Dow Theory.")
    else:
        trends.append("The stock is in a sideways trend according to Dow Theory.")
    
    return " ".join(trends)

# Plotting function
def plot_data(data, model):
    features = prepare_features(data)
    
    if features.shape[0] > 0:
        predictions = model.predict(features[['Pct_Change', 'Volatility']])
        data = data.loc[features.index]  # Align the indices
        data['Prediction'] = predictions
        
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
st.title('Market Analysis with Machine Learning and Technical Indicators')

ticker = st.text_input('Enter stock ticker (e.g., AAPL):', 'AAPL')
start_date = st.date_input('Start date', pd.to_datetime('2020-01-01'))
end_date = st.date_input('End date', pd.to_datetime('today'))

if st.button('Analyze'):
    with st.spinner('Fetching data...'):
        data = fetch_data(ticker, start_date, end_date)
        
        if data.empty:
            st.error("No data found for this ticker.")
        else:
            data = calculate_technical_indicators(data)  # Add technical indicators
            features = prepare_features(data)
            if features.shape[0] == 0:
                st.error("Not enough data to prepare features.")
            else:
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

                # Generate advisory
                advisory = generate_advisory(data, model)
                st.write("### Advisory")
                st.text(advisory)

                # Plot the data
                plot_data(data, model)

# Note: Ensure to run this code in an environment with the required libraries.
