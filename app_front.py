import os
os.environ["YFINANCE_CACHE_DIR"] = "/tmp/yfinance_cache"  # Set a temporary cache directory
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

if not os.path.exists(os.environ["YFINANCE_CACHE_DIR"]):
    os.makedirs(os.environ["YFINANCE_CACHE_DIR"])

# Load the pre-trained BiLSTM model
model = tf.keras.models.load_model("C:/Users/HP/bilstm_stock_model.h5")  # Adjust the path to your saved model

# Function to fetch data
def fetch_data(stock_symbol, start_date='2010-01-01', end_date='2024-12-31'):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Function to preprocess data
def preprocess_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    last_time_step_data = scaled_data[-time_step:]  # Last `time_step` data for prediction
    return scaled_data, last_time_step_data, scaler

# Function to predict future values
def predict_future(stock_symbol, forecast_days=10, time_step=60):
    # Fetch and preprocess data
    data = fetch_data(stock_symbol)
    _, last_time_step_data, scaler = preprocess_data(data, time_step)
    
    # Reshape the last_time_step_data for model input
    last_data = np.reshape(last_time_step_data, (1, last_time_step_data.shape[0], 1))
    future_predictions = []

    # Predict for the specified number of days
    for _ in range(forecast_days):
        predicted_price = model.predict(last_data)[0][0]
        future_predictions.append(predicted_price)
        
        # Update last_data with the new prediction
        last_data = np.append(last_data[0][1:], [[predicted_price]], axis=0)
        last_data = np.reshape(last_data, (1, last_data.shape[0], 1))

    # Transform predictions back to original scale
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions, data

# Streamlit App
st.title("Stock Price Prediction using BiLSTM")

# User inputs
st.sidebar.header("User Input Parameters")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
forecast_days = st.sidebar.slider("Number of Days to Predict", min_value=1, max_value=30, value=10)

# Predict and Display Results
if st.sidebar.button("Predict"):
    st.write(f"Fetching data and predicting future prices for {stock_symbol}...")
    
    try:
        future_predictions, historical_data = predict_future(stock_symbol, forecast_days)
        
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(historical_data['Close'], label=f"{stock_symbol} Historical Prices", color="blue")
        ax.plot(
            range(len(historical_data), len(historical_data) + forecast_days),
            future_predictions,
            label=f"{stock_symbol} Future Predictions",
            color="orange",
        )
        ax.set_title(f"{stock_symbol} Stock Price Prediction")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
        
        # Display predicted values
        st.write(f"Predicted Prices for the next {forecast_days} days:")
        future_df = pd.DataFrame(
            {
                "Day": [f"Day {i+1}" for i in range(forecast_days)],
                "Predicted Price": future_predictions.flatten(),
            }
        )
        st.write(future_df)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
