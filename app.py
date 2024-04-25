import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

# Feature Engineering
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Prepare Data for Training
X = data[['Year', 'Month', 'Day', 'Open', 'High', 'Low', 'Close']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Forecast using Random Forest
st.subheader('Forecast using Random Forest')

# Predict on Test Data
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.write('Root Mean Squared Error:', rmse)

# Plot forecast
plt.figure(figsize=(10, 6))
plt.plot(data['Date'][-len(y_test):], y_test, label='Actual')
plt.plot(data['Date'][-len(y_test):], y_pred, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Forecast using Random Forest')
plt.legend()
st.pyplot()
