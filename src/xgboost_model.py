# src/xgboost_model.py

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Add path to evaluation utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from evaluate_models import evaluate_forecast, print_evaluation

# === CONFIG DEFAULTS ===
DEFAULT_TICKER = 'AAPL'
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'xgboost'))

warnings.filterwarnings('ignore')

# === FUNCTIONS ===

def load_and_prepare_data(filepath, date_col, value_col):
    df = pd.read_csv(filepath, header=1)
    df = df[df[date_col] != 'Date'].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = df.asfreq('B')

    if value_col not in df.columns:
        raise ValueError(f"❌ Column '{value_col}' not found in '{filepath}'.")

    df['Close'] = pd.to_numeric(df[value_col], errors='coerce')
    df['Close'].interpolate(method='linear', inplace=True)
    return df[['Close']].copy()

def create_features(df):
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    df.dropna(inplace=True)
    return df

def plot_forecast(test_index, actual, forecast, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(test_index, actual, label='Actual')
    plt.plot(test_index, forecast, label='Forecast')
    plt.title('XGBoost Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📸 Forecast plot saved to {save_path}")

def save_forecast(test_index, forecast_values, save_path):
    forecast_df = pd.DataFrame({
        'Date': test_index,
        'Forecast': forecast_values
    })
    forecast_df.to_csv(save_path, index=False)
    print(f"📄 Forecast values saved to {save_path}")

def save_evaluation(results_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("\n📊 Evaluation for XGBoost\n")
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
    df = create_features(df)

    print("🔄 Splitting data and scaling...")
    split_index = len(df) - 30
    train, test = df.iloc[:split_index], df.iloc[split_index:]

    features = ['dayofweek', 'month'] + [f'lag_{i}' for i in range(1, 6)]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    y_train = train['Close'].values
    X_test = scaler.transform(test[features])
    y_test = test['Close'].values

    print("🧠 Training XGBoost model...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror')
    model.fit(X_train, y_train)

    print("📈 Forecasting...")
    forecast = model.predict(X_test)

    print("✅ Evaluating forecast...")
    results = evaluate_forecast(y_test, forecast, model_name=f'XGBoost_{ticker}')
    print_evaluation(results)
    save_evaluation(results, save_path=results_path)
    save_forecast(test.index, forecast, forecast_path)
    plot_forecast(test.index, y_test, forecast, save_path=plot_path)

# === CLI SUPPORT ===

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run XGBoost Forecasting Engine for a specific ticker.')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol, e.g., AAPL')
    parser.add_argument('--date_col', type=str, default='Ticker', help='Column containing dates')
    parser.add_argument('--value_col', type=str, default='', help='Column containing target values (optional)')
    args = parser.parse_args()
    main(ticker=args.ticker, date_col=args.date_col, value_col=args.value_col)
