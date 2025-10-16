# Stock Price Prediction using Python
# Author: Your Name
# Last Updated: Oct 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# 1. Fetch Stock Data
# ==============================

ticker = "AAPL"  # Change to any stock symbol (e.g., TSLA, MSFT)
data = yf.download(ticker, start="2020-01-01", end="2025-10-01")

print("\n✅ Data Downloaded Successfully!\n")
print(data.head())

# ==============================
# 2. Feature Engineering
# ==============================

data['Prev_Close'] = data['Close'].shift(1)
data['Price_Change'] = data['Close'] - data['Prev_Close']
data.dropna(inplace=True)

X = data[['Prev_Close']]
y = data['Close']

# ==============================
# 3. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# 4. Model Training
# ==============================

model = LinearRegression()
model.fit(X_train, y_train)

print("\n🤖 Model Training Complete!\n")

# ==============================
# 5. Evaluation
# ==============================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Evaluation Metrics:")
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))
print("R²:", round(r2, 3))

# ==============================
# 6. Plot Actual vs Predicted
# ==============================

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual Price", color="blue")
plt.plot(y_pred, label="Predicted Price", color="red", alpha=0.7)
plt.title(f"Actual vs Predicted Prices - {ticker}")
plt.xlabel("Samples")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

# ==============================
# 7. Future Prediction Example
# ==============================

last_price = data['Close'].iloc[-1]
predicted_next = model.predict([[last_price]])
print(f"\n💰 Predicted Next Closing Price for {ticker}: ${predicted_next[0]:.2f}")
