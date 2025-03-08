import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import requests
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# Step 1: Fetch Historical Gold Price Data from API
url = 'https://www.chards.co.uk/data/update-price-chart'
params = {
    'MetalPricesForm[metal]': 'gold',
    'MetalPricesForm[weight]': 'kilogram',
    'MetalPricesForm[currency]': 'usd',
    'MetalPricesForm[period]': 'all-time',
    'manual': 'true'
}
response = requests.get(url, params=params)
data = response.json()

# Extract and clean data
dates = pd.to_datetime(data['chart_data']['labels'])
prices = data['chart_data']['values']
df = pd.DataFrame({'Date': dates, 'Gold_Price': prices})
df.set_index('Date', inplace=True)
df = df.dropna()

# Step 2: Stationarity Test Function
def adf_test(series):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f'ADF Test Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    return result[1] <= 0.05  # Returns True if stationary

# Step 3: Make Series Stationary if Needed
df['Gold_Price_Diff1'] = df['Gold_Price'].diff()
df['Gold_Price_Diff2'] = df['Gold_Price_Diff1'].diff()

if not adf_test(df['Gold_Price']):
    if adf_test(df['Gold_Price_Diff1']):
        df['Stationary'] = df['Gold_Price_Diff1']
        d = 1
    elif adf_test(df['Gold_Price_Diff2']):
        df['Stationary'] = df['Gold_Price_Diff2']
        d = 2
    else:
        raise ValueError("Data is still non-stationary after differencing")
else:
    df['Stationary'] = df['Gold_Price']
    d = 0

# Step 4: Train/Test Split
train_size = int(len(df) * 0.6)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Step 5: Determine ARIMA Parameters using Auto-ARIMA
auto_arima_model = pm.auto_arima(train['Stationary'].dropna(), d=1, seasonal=False, trace=True)
p, _, q = auto_arima_model.order

# Step 6: Train ARIMA Model with Trend Component
model = ARIMA(train['Gold_Price'], order=(p, d, q), trend='t')
fitted_model = model.fit()

# Step 7: Walk-Forward Validation
history = list(train['Gold_Price'])
predictions = []
for t in range(len(test)):
    model = ARIMA(history, order=(p, d, q), trend='t')
    fitted_model = model.fit()
    yhat = fitted_model.forecast()[0]
    predictions.append(yhat)
    history.append(test['Gold_Price'].iloc[t])

# Step 8: Evaluate Model
mae = mean_absolute_error(test['Gold_Price'], predictions)
mse = mean_squared_error(test['Gold_Price'], predictions)
rmse = np.sqrt(mse)
print(f"MAE: {mae}, RMSE: {rmse}")

# Thêm MAPE
def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

mape = calculate_mape(test['Gold_Price'], predictions)
print(f"MAPE: {mape:.2f}%")

# Phân tích phần dư
residuals = test['Gold_Price'] - predictions
plt.figure(figsize=(14, 4))
plt.plot(test.index, residuals, label='Residuals', color='purple')
plt.axhline(0, linestyle='--', color='black')
plt.title('Residuals of ARIMA Model')
plt.xlabel('Date')
plt.ylabel('Residual Value')
plt.legend()
plt.grid()
plt.show()

plot_acf(residuals, lags=20)
plt.title('ACF of Residuals')
plt.show()

print(f"AIC: {fitted_model.aic}")
print(f"BIC: {fitted_model.bic}")

# Step 9: Forecast Future Prices
forecast_steps = 300  # Thêm định nghĩa forecast_steps
future_forecast = fitted_model.forecast(steps=forecast_steps)
future_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='B')[1:]
forecast_df = pd.DataFrame({'Forecast': future_forecast}, index=future_dates)

# Step 10: Visualization
# So sánh trên tập test
plt.figure(figsize=(14, 6))
plt.plot(test.index, test['Gold_Price'], label='Actual Price (Test)', color='blue')
plt.plot(test.index, predictions, label='Predicted Price (Test)', color='orange', linestyle='--')
plt.legend()
plt.title('Actual vs Predicted Gold Prices (Test Set)')
plt.xlabel('Date')
plt.ylabel('Gold Price (USD)')
plt.grid()
plt.show()

# So sánh trên toàn bộ dữ liệu
full_model = ARIMA(df['Gold_Price'], order=(p, d, q), trend='t')
full_fitted = full_model.fit()
full_predictions = full_fitted.predict(start=df.index[0], end=df.index[-1], typ='levels')

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Gold_Price'], label='Actual Price', color='blue')
plt.plot(df.index, full_predictions, label='Fitted Values', color='orange', linestyle='--')
plt.legend()
plt.title('Actual vs Fitted Gold Prices (Full Dataset)')
plt.xlabel('Date')
plt.ylabel('Gold Price (USD)')
plt.grid()
plt.show()

# Dự báo với khoảng tin cậy
forecast_obj = fitted_model.get_forecast(steps=forecast_steps)
forecast = forecast_obj.predicted_mean
conf_int = forecast_obj.conf_int(alpha=0.05)

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Gold_Price'], label='Actual Price', color='blue')
plt.plot(test.index, predictions, label='Walk-Forward Prediction', color='orange')
plt.plot(forecast_df.index, forecast, label='Future Forecast', color='red', linestyle='dashed')
plt.fill_between(forecast_df.index, conf_int[:, 0], conf_int[:, 1],  # Sửa từ .iloc sang chỉ số NumPy
                 color='red', alpha=0.2, label='95% Confidence Interval')
plt.legend()
plt.title('Gold Price Forecast with Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Gold Price (USD)')
plt.grid()
plt.show()
