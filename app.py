import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Download data
start = '2012-01-01'
end = '2024-01-10'
stock = 'CSCO'
data = yf.download(stock, start, end)
data.reset_index(inplace=True)

# Calculate moving averages
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

# Plot moving averages and closing price
plt.figure(figsize=(10, 6))
plt.plot(ma_100_days, 'r', label='MA 100 days')
plt.plot(ma_200_days, 'b', label='MA 200 days')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
plt.show()

# Prepare data for Random Forest
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.Close.values.reshape(-1, 1))

x, y = [], []
for i in range(100, len(data_scaled)):
    x.append(data_scaled[i-100:i, 0])
    y.append(data_scaled[i, 0])
x, y = np.array(x), np.array(y)

# Split data into train and test sets
split_index = int(len(data) * 0.8)
x_train, y_train = x[:split_index], y[:split_index]
x_test, y_test = x[split_index:], y[split_index:]

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate the model
y_pred_train = model.predict(x_train)
mse_train = mean_squared_error(y_train, y_pred_train)
print('Train MSE:', mse_train)

y_pred_test = model.predict(x_test)
mse_test = mean_squared_error(y_test, y_pred_test)
print('Test MSE:', mse_test)

# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test, color='blue', label='Actual Stock Price')
plt.plot(y_pred_test, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction (Random Forest)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
