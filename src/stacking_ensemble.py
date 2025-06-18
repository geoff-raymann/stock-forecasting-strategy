# src/stacking_ensemble.py

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from xgboost import XGBRegressor

# Add path to evaluation module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from evaluate_models import evaluate_forecast, print_evaluation

# === CONFIG ===
DEFAULT_TICKER = 'AAPL'
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'ensemble'))

SEQUENCE_LENGTH = 60
EPOCHS = 50
BATCH_SIZE = 16

# === DATA FUNCTIONS ===
def load_data(filepath, date_col, value_col):
    df = pd.read_csv(filepath, header=1)
    df = df[df[date_col] != 'Date']
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = df.asfreq('B')
    df['Close'] = pd.to_numeric(df[value_col], errors='coerce')
    df['Close'].interpolate(method='linear', inplace=True)
    return df[['Close']]

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# === LSTM MODEL ===
def train_lstm(X_train, y_train):
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[EarlyStopping(patience=5)])
    return model

# === XGBOOST MODEL ===
def train_xgboost(X_train, y_train):
    model = XGBRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# === PLOTTING ===
def plot_forecast(actual, forecast, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual.values, label='Actual')
    plt.plot(actual.index, forecast, label='Stacked Forecast')
    plt.title('Stacking Ensemble Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📸 Forecast plot saved to {save_path}")

# === MAIN ===
def main(ticker=DEFAULT_TICKER, date_col='Ticker', value_col='AAPL.3'):
    file_path = os.path.join(DEFAULT_DATA_DIR, f'{ticker}.csv')
    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, ticker)
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, f'{ticker.lower()}_stacking_plot.png')
    results_path = os.path.join(output_dir, f'{ticker.lower()}_stacking_eval.txt')
    forecast_path = os.path.join(output_dir, f'{ticker.lower()}_stacking_forecast.csv')

    print(f"📥 Loading and preparing data for {ticker}...")
    df = load_data(file_path, date_col, value_col)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

    split_index = len(df) - 30
    X_train, X_test = X[:split_index - SEQUENCE_LENGTH], X[split_index - SEQUENCE_LENGTH:]
    y_train, y_test = y[:split_index - SEQUENCE_LENGTH], y[split_index - SEQUENCE_LENGTH:]

    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print("🧠 Training base models...")
    lstm_model = train_lstm(X_train_lstm, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    print("🔄 Generating base model forecasts...")
    lstm_preds = lstm_model.predict(X_test_lstm)
    xgb_preds = xgb_model.predict(X_test)

    meta_X = np.vstack([lstm_preds.flatten(), xgb_preds.flatten()]).T
    meta_y = y_test

    print("🔧 Training meta-model (Ridge)...")
    ridge = Ridge()
    ridge.fit(meta_X, meta_y)
    stacked_preds = ridge.predict(meta_X)

    forecast = scaler.inverse_transform(stacked_preds.reshape(-1, 1))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    print("✅ Evaluating forecast...")
    results = evaluate_forecast(actual.flatten(), forecast.flatten(), model_name=f'Ensemble_{ticker}')
    print_evaluation(results)

    pd.DataFrame({
        'Date': df.index[-30:],
        'Forecast': forecast.flatten()
    }).to_csv(forecast_path, index=False)

    plot_forecast(df[-30:], forecast, save_path=plot_path)

    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("📊 Evaluation for Stacking Ensemble\n")
        f.write(f"{'-'*30}\n")
        for k, v in results.items():
            try:
                f.write(f"{k.upper():<6}: {float(v):.4f}\n")
            except (ValueError, TypeError):
                f.write(f"{k.upper():<6}: {v}\n")
        f.write(f"\nSaved on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# === CLI ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run stacking ensemble for time series forecast.')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol')
    parser.add_argument('--date_col', type=str, default='Ticker', help='Date column name')
    parser.add_argument('--value_col', type=str, default='AAPL.3', help='Target value column')
    args = parser.parse_args()

    main(ticker=args.ticker, date_col=args.date_col, value_col=args.value_col)
