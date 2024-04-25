import streamlit as st
import yfinance as yf
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Load or fetch S&P 500 data
if os.path.exists("sp500.csv"):
    sp500 = pd.read_csv("sp500.csv", index_col=0)
else:
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    sp500.to_csv("sp500.csv")

sp500.index = pd.to_datetime(sp500.index)

# Streamlit app
st.title('S&P 500 Stock Prediction')

# Display the data
st.subheader('S&P 500 Data')
st.write(sp500)

# Plot Close price
st.subheader('S&P 500 Close Price')
st.line_chart(sp500['Close'])

# Data preprocessing
del sp500["Dividends"]
del sp500["Stock Splits"]
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

# Define predictors
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Train and predict function
@st.cache
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Backtesting function
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)

# Perform backtesting
predictions = backtest(sp500, model, predictors)

# Display predictions
st.subheader('Predictions')
st.write(predictions)

# Calculate precision score
precision = precision_score(predictions["Target"], predictions["Predictions"])
st.write('Precision Score:', precision)

