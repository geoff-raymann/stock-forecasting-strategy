# src/prophet_model.py

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add evaluation module to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from evaluate_models import evaluate_forecast, print_evaluation

# === CONFIG ===
DEFAULT_TICKER = 'AAPL'
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'prophet'))

warnings.filterwarnings("ignore")

# === FUNCTIONS ===

def load_and_prepare_data(filepath, date_col, value_col):
    df = pd.read_csv(filepath, header=1)
    df = df[df[date_col] != 'Date'].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df[[date_col, value_col]].dropna()
    df.columns = ['ds', 'y']
    df = df.set_index('ds').asfreq('B').reset_index()
    df['y'].interpolate(method='linear', inplace=True)
    return df

def plot_forecast(df, forecast, ticker, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df['y'], label='Actual')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3)
    plt.title(f"Prophet Forecast vs Actual for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📸 Forecast plot saved to {save_path}")

def save_forecast(forecast_df, save_path):
    forecast_df[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Forecast'}).to_csv(save_path, index=False)
    print(f"📄 Forecast values saved to {save_path}")


def save_evaluation(results_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("📊 Evaluation for Prophet\n")
        f.write(f"{'-'*30}\n")
        for metric, value in results_dict.items():
            if isinstance(value, (int, float)):
                f.write(f"{metric.upper():<6}: {value:.4f}\n")
            else:
                f.write(f"{metric.upper():<6}: {value}\n")
        f.write(f"\nSaved on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"📄 Evaluation results saved to {save_path}")

# === MAIN SCRIPT ===

def main(ticker=DEFAULT_TICKER, date_col='Ticker', value_col='AAPL.3'):
    file_path = os.path.join(DEFAULT_DATA_DIR, f'{ticker}.csv')
    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, ticker)
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, f'{ticker.lower()}_forecast_plot.png')
    results_path = os.path.join(output_dir, f'{ticker.lower()}_evaluation.txt')
    forecast_path = os.path.join(output_dir, f'{ticker.lower()}_forecast.csv')

    print(f"📥 Loading and preparing data for {ticker}...")
    df = load_and_prepare_data(file_path, date_col, value_col)

    print("✂️ Splitting into train and test...")
    train = df[:-30].copy()
    test = df[-30:].copy()

    print("🧠 Fitting Prophet model...")
    model = Prophet()
    model.fit(train)

    print("🔮 Making future forecast...")
    future = model.make_future_dataframe(periods=30, freq='B')
    forecast = model.predict(future)

    print("📈 Plotting forecast...")
    plot_forecast(df, forecast, ticker, save_path=plot_path)

    print("✅ Evaluating forecast...")
    forecast_subset = forecast.set_index('ds').loc[test['ds']]
    predicted = forecast_subset['yhat'].values
    actual = test['y'].values

    results = evaluate_forecast(actual, predicted, model_name=f'Prophet_{ticker}')
    print_evaluation(results)
    save_evaluation(results, save_path=results_path)

    print("💾 Saving forecast...")
    save_forecast(forecast, forecast_path)
    print(f"✅ All Prophet results saved for {ticker}")


# === CLI SUPPORT ===

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Prophet Forecasting Engine for a specific ticker.')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol, e.g., AAPL')
    parser.add_argument('--date_col', type=str, default='Ticker', help='Column containing dates')
    parser.add_argument('--value_col', type=str, default='AAPL.3', help='Column containing target values')
    args = parser.parse_args()
    main(ticker=args.ticker, date_col=args.date_col, value_col=args.value_col)