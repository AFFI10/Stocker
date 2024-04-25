import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import load_model

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

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.Close.values.reshape(-1, 1))

x, y = [], []
for i in range(100, len(data_scaled)):
    x.append(data_scaled[i-100:i, 0])
    y.append(data_scaled[i, 0])
x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Define LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x, y, epochs=50, batch_size=32, verbose=1)

# Save the model
model.save('afroze')

# Load the model
model = load_model('afroze')

# Split data into train and test sets
split_index = int(len(data) * 0.8)
data_train = data.Close[:split_index]
data_test = data.Close[split_index:]

# Scale the test data
data_test_scaled = scaler.transform(data_test.values.reshape(-1, 1))

# Prepare test data
x_test, y_test = [], []
for i in range(100, len(data_test_scaled)):
    x_test.append(data_test_scaled[i-100:i, 0])
    y_test.append(data_test_scaled[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Evaluate the model
scores = model.evaluate(x_test, y_test)
LSTM_accuracy = scores * 100
print('Test accuracy:', LSTM_accuracy, '%')

# Predictions
y_predict = model.predict(x_test)

# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test, color='blue', label='Actual Stock Price')
plt.plot(y_predict, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
