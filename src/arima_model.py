# src/arima_model.py

import os
import sys
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Add path to evaluate_models.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from evaluate_models import evaluate_forecast, print_evaluation

# Config
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'AAPL.csv'))
DATE_COL = 'Ticker'           # The column with dates
CLOSE_COL = 'AAPL.3'          # The actual closing price column to use

def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH, header=1)

    # Remove the duplicated header row
    df = df[df[DATE_COL] != 'Date'].copy()

    # Convert date and set index
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df.set_index(DATE_COL, inplace=True)
    df = df.asfreq('B')

    # Extract Close prices
    df['Close'] = pd.to_numeric(df[CLOSE_COL], errors='coerce')
    df['Close'].interpolate(method='linear', inplace=True)

    return df[['Close']].copy()

def test_stationarity(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")

def plot_series(train, test, forecast):
    plt.figure(figsize=(12, 6))
    plt.plot(train[-60:], label='Train')
    plt.plot(test, label='Actual')
    plt.plot(forecast, label='Forecast')
    plt.title('ARIMA Forecast vs Actual')
    plt.legend()
    plt.show()

def main():
    warnings.filterwarnings('ignore')

    print("📥 Loading and preparing data...")
    df = load_and_prepare_data()

    print("📊 Testing stationarity...")
    test_stationarity(df['Close'])

    print("🔧 Fitting ARIMA model...")
    train = df['Close'][:-30]
    test = df['Close'][-30:]

    model = ARIMA(train, order=(5, 1, 2))
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=30)

    print("📈 Plotting forecast vs actual...")
    plot_series(train, test, forecast)

    print("✅ Evaluating forecast...")
    results = evaluate_forecast(test.values, forecast.values, model_name='ARIMA')
    print_evaluation(results)

if __name__ == "__main__":
    main()
