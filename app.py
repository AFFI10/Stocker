import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Function to download and preprocess data
def load_data(stock, start, end):
    data = yf.download(stock, start, end)
    data.reset_index(inplace=True)
    return data

# Function to calculate moving averages
def calculate_moving_averages(data):
    ma_100_days = data.Close.rolling(100).mean()
    ma_200_days = data.Close.rolling(200).mean()
    return ma_100_days, ma_200_days

# Function to prepare data for Random Forest
def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.Close.values.reshape(-1, 1))

    x, y = [], []
    for i in range(100, len(data_scaled)):
        x.append(data_scaled[i-100:i, 0])
        y.append(data_scaled[i, 0])
    x, y = np.array(x), np.array(y)
    return x, y

# Function to train Random Forest model
def train_model(x_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model

# Function to evaluate model
def evaluate_model(model, x_train, y_train, x_test, y_test):
    y_pred_train = model.predict(x_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    
    y_pred_test = model.predict(x_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    return mse_train, mse_test, y_test, y_pred_test

# Function to plot predictions
def plot_predictions(y_test, y_pred_test):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, color='blue', label='Actual Stock Price')
    plt.plot(y_pred_test, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction (Random Forest)')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot()

# Main function to run Streamlit app
def main():
    st.title('Stock Price Prediction with Random Forest')
    st.sidebar.title('Options')

    stock = st.sidebar.text_input('Enter Stock Ticker', 'CSCO')
    start = st.sidebar.text_input('Start Date', '2012-01-01')
    end = st.sidebar.text_input('End Date', '2024-01-10')

    data = load_data(stock, start, end)
    ma_100_days, ma_200_days = calculate_moving_averages(data)

    st.subheader('Moving Averages')
    plt.figure(figsize=(10, 6))
    plt.plot(ma_100_days, 'r', label='MA 100 days')
    plt.plot(ma_200_days, 'b', label='MA 200 days')
    plt.plot(data.Close, 'g', label='Closing Price')
    plt.legend()
    st.pyplot()

    x, y = prepare_data(data)
    split_index = int(len(data) * 0.8)
    x_train, y_train = x[:split_index], y[:split_index]
    x_test, y_test = x[split_index:], y[split_index:]

    model = train_model(x_train, y_train)

    mse_train, mse_test, y_test, y_pred_test = evaluate_model(model, x_train, y_train, x_test, y_test)
    st.write(f'Train MSE: {mse_train}')
    st.write(f'Test MSE: {mse_test}')

    plot_predictions(y_test, y_pred_test)

if __name__ == '__main__':
    main()
