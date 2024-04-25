import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Data
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
st.subheader('Time Series data with Rangeslider')
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Open'], label='Open')
plt.plot(data['Date'], data['Close'], label='Close')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Prices Over Time')
plt.legend()
st.pyplot()

# Forecast using Simple Moving Average (SMA)
st.subheader('Forecast using Simple Moving Average (SMA)')

# Prepare data
close_prices = data['Close'].values
window_size = 30  # You can adjust this parameter as needed
sma_forecast = []

# Calculate SMA
for i in range(len(close_prices) - window_size):
    sma = np.mean(close_prices[i:i + window_size])
    sma_forecast.append(sma)

# Pad the forecast with NaNs to align with the original data
sma_forecast = np.concatenate((np.full(window_size, np.nan), sma_forecast))

# Plot forecast
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Close'], label='Actual')
plt.plot(data['Date'], sma_forecast, label='SMA Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Forecast using Simple Moving Average (SMA)')
plt.legend()
st.pyplot()

# Predicted values
predicted_values = sma_forecast[-period:]

# Plot predicted values
st.subheader('Predicted Values')
plt.figure(figsize=(10, 6))
plt.plot(data['Date'][-period:], data['Close'][-period:], label='Actual')
plt.plot(data['Date'][-period:], predicted_values, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Predicted Stock Prices')
plt.legend()
st.pyplot()
# Plot bar graph for predicted stock price
st.subheader('Predicted Stock Price for the Next {} Years'.format(n_years))
if not data.empty:
    plt.figure(figsize=(8, 6))
    plt.bar('Current', data['Close'][-1], color='blue', label='Current Price')
    plt.bar('Predicted', predicted_values, color='orange', label='Predicted Price')
    plt.xlabel('Price')
    plt.ylabel('Stock Price')
    plt.title('Predicted Stock Price for {} in the Next {} Years'.format(selected_stock, n_years))
    plt.legend()
    st.pyplot()
else:
    st.warning("Data is empty. Please enter a valid stock symbol.")




# Display additional information
st.sidebar.subheader('About')
st.sidebar.info('This app is for educational purposes only. Use at your own risk!')

# Display contact information
st.sidebar.subheader('Contact Information')
st.sidebar.text('Developer: Disha Barmola')

