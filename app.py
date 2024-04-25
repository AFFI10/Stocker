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
