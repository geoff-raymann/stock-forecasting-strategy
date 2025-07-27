# src/arima_model.py

import os
import sys
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from datetime import datetime

# Add path to evaluation module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from evaluate_models import evaluate_forecast, print_evaluation

# === CONFIG DEFAULTS ===
DEFAULT_TICKER = 'AAPL'
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'arima'))

# === FUNCTIONS ===

def load_and_prepare_data(filepath, date_col, close_col):
    df = pd.read_csv(filepath, header=1)
    df = df[df[date_col] != 'Date'].copy()  # Remove second header row
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = df.asfreq('B')  # Set business day frequency

    if close_col not in df.columns:
        raise ValueError(f"❌ Column '{close_col}' not found in file '{filepath}'.")

    df['Close'] = pd.to_numeric(df[close_col], errors='coerce')
    df['Close'].interpolate(method='linear', inplace=True)
    return df[['Close']].copy()

def test_stationarity(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    return result[0], result[1]

def plot_series(train, test, forecast, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(train[-60:], label='Train')
    plt.plot(test, label='Actual')
    plt.plot(test.index, forecast, label='Forecast')
    plt.title('ARIMA Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📸 Forecast plot saved to {save_path}")

def save_evaluation(results_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("📊 Evaluation for ARIMA\n")
        f.write(f"{'-'*30}\n")
        for metric, value in results_dict.items():
            if isinstance(value, (int, float)):
                f.write(f"{metric.upper():<6}: {value:.4f}\n")
            else:
                f.write(f"{metric.upper():<6}: {value}\n")
        f.write(f"\nSaved on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"📄 Evaluation results saved to {save_path}")

def save_forecast(test_index, forecast_values, save_path):
    forecast_df = pd.DataFrame({
        'Date': test_index,
        'Forecast': forecast_values
    })
    forecast_df.to_csv(save_path, index=False)
    print(f"📄 Forecast values saved to {save_path}")

# === MAIN SCRIPT ===

def main(ticker=DEFAULT_TICKER, date_col='Ticker', close_col=None):
    warnings.filterwarnings('ignore')
    if close_col is None or close_col.strip() == "":
        close_col = f'{ticker}.3'

    # Paths
    file_path = os.path.join(DEFAULT_DATA_DIR, f'{ticker}.csv')
    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, ticker)
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, f'{ticker.lower()}_forecast_plot.png')
    results_path = os.path.join(output_dir, f'{ticker.lower()}_evaluation.txt')
    forecast_path = os.path.join(output_dir, f'{ticker.lower()}_forecast.csv')

    print(f"📥 Loading and preparing data for {ticker}...")
    df = load_and_prepare_data(file_path, date_col, close_col)

    print("📊 Testing stationarity...")
    test_stationarity(df['Close'])

    print("🔧 Fitting ARIMA model...")
    train = df['Close'][:-30]
    test = df['Close'][-30:]
    model = ARIMA(train, order=(5, 1, 2))
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=30)

    print("📈 Plotting forecast vs actual...")
    plot_series(train, test, forecast, save_path=plot_path)

    print("✅ Evaluating forecast...")
    results = evaluate_forecast(test.values, forecast.values, model_name=f'ARIMA_{ticker}')
    print_evaluation(results)
    save_evaluation(results, save_path=results_path)

    print("💾 Saving forecast values...")
    save_forecast(test.index, forecast, save_path=forecast_path)

# === CLI SUPPORT ===

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run ARIMA Forecasting Engine for a specific ticker.')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol, e.g., AAPL')
    parser.add_argument('--date_col', type=str, default='Ticker', help='Column containing dates')
    parser.add_argument('--close_col', type=str, default='', help='Column containing closing prices (optional)')

    args = parser.parse_args()
    main(ticker=args.ticker, date_col=args.date_col, close_col=args.close_col)
