# src/lstm_model.py

import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

# Add path to evaluation module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from evaluate_models import evaluate_forecast, print_evaluation

# === CONFIG DEFAULTS ===
DEFAULT_TICKER = 'AAPL'
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'lstm'))

SEQUENCE_LENGTH = 60
EPOCHS = 50
BATCH_SIZE = 16

warnings.filterwarnings('ignore')

# === FUNCTIONS ===

def load_and_prepare_data(filepath, date_col, value_col):
    df = pd.read_csv(filepath, header=1)
    df = df[df[date_col] != 'Date'].copy()  # Remove second header row
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = df.asfreq('B')

    if value_col not in df.columns:
        raise ValueError(f"❌ Column '{value_col}' not found in '{filepath}'")

    df['Close'] = pd.to_numeric(df[value_col], errors='coerce')
    df['Close'].interpolate(method='linear', inplace=True)
    return df[['Close']].copy()

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_forecast(test, forecast, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test.values, label='Actual')
    plt.plot(test.index, forecast, label='Forecast')
    plt.title('LSTM Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📸 Forecast plot saved to {save_path}")

def save_forecast(test_index, forecast_values, save_path, actual_values=None):
    forecast_df = pd.DataFrame({
        'Date': test_index,
        'Forecast': forecast_values
    })
    if actual_values is not None:
        forecast_df['Actual'] = actual_values
    forecast_df.to_csv(save_path, index=False)
    print(f"📄 Forecast values saved to {save_path}")

def save_evaluation(results_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("📊 Evaluation for LSTM\n")
        f.write(f"{'-'*30}\n")
        for metric, value in results_dict.items():
            if isinstance(value, (int, float)):
                f.write(f"{metric.upper():<6}: {value:.4f}\n")
            else:
                f.write(f"{metric.upper():<6}: {value}\n")
        f.write(f"\nSaved on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"📄 Evaluation results saved to {save_path}")

# === MAIN SCRIPT ===

def main(ticker=DEFAULT_TICKER, date_col='Ticker', value_col=None):
    if not value_col or value_col.strip() == "":
        value_col = f'{ticker}.3'

    file_path = os.path.join(DEFAULT_DATA_DIR, f'{ticker}.csv')
    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, ticker)
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, f'{ticker.lower()}_forecast_plot.png')
    results_path = os.path.join(output_dir, f'{ticker.lower()}_evaluation.txt')
    forecast_path = os.path.join(output_dir, f'{ticker.lower()}_forecast.csv')

    print(f"📥 Loading and preparing data for {ticker}...")
    df = load_and_prepare_data(file_path, date_col, value_col)

    print("🔄 Scaling data and creating sequences...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

    split_index = len(df) - 30
    X_train, X_test = X[:split_index - SEQUENCE_LENGTH], X[split_index - SEQUENCE_LENGTH:]
    y_train, y_test = y[:split_index - SEQUENCE_LENGTH], y[split_index - SEQUENCE_LENGTH:]

    print("🧠 Building and training model...")
    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[EarlyStopping(patience=5)])

    print("📈 Forecasting...")
    predictions = model.predict(X_test)
    forecast = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    test_index = df.index[-30:]
    plot_forecast(df[-30:], forecast, save_path=plot_path)

    print("✅ Evaluating forecast...")
    results = evaluate_forecast(actual.flatten(), forecast.flatten(), model_name=f'LSTM_{ticker}')
    print_evaluation(results)
    save_evaluation(results, save_path=results_path)
    save_forecast(test_index, forecast.flatten(), save_path=forecast_path, actual_values=actual.flatten())

# === CLI SUPPORT ===

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LSTM Forecasting Engine for a specific ticker.')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol, e.g., AAPL')
    parser.add_argument('--date_col', type=str, default='Ticker', help='Column containing dates')
    parser.add_argument('--value_col', type=str, default='', help='Column containing target values (optional)')
    args = parser.parse_args()
    main(ticker=args.ticker, date_col=args.date_col, value_col=args.value_col)
