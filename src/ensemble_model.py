# src/ensemble_model.py

import os
import argparse
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from evaluate_models import evaluate_forecast, print_evaluation

# === UTILITY FUNCTIONS ===

def load_forecast(path):
    df = pd.read_csv(path)
    # Handle Prophet output
    if 'ds' in df.columns and 'yhat' in df.columns:
        df = df.rename(columns={'ds': 'Date', 'yhat': 'Forecast'})
    if 'Date' not in df.columns:
        raise KeyError(f"'Date' column not found in {path}. Columns found: {list(df.columns)}")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df
def load_actual(path, value_col='AAPL.3', date_col='Ticker'):
    df = pd.read_csv(path, header=1)
    df = df[df[date_col] != 'Date']
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = df.asfreq('B')
    df['Close'] = pd.to_numeric(df[value_col], errors='coerce')
    df['Close'] = df['Close'].interpolate(method='linear')
    return df['Close']

def save_forecast(df, path):
    df.to_csv(path)
    print(f"✅ Ensemble forecast saved to {path}")

def save_evaluation(results, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("📊 Evaluation for Ensemble\n")
        f.write(f"{'-'*30}\n")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                f.write(f"{metric.upper():<6}: {value:.4f}\n")
            else:
                f.write(f"{metric.upper():<6}: {value}\n")
        f.write(f"\nSaved on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"📄 Evaluation results saved to {path}")

def plot_ensemble(actual, forecast, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual.values, label='Actual', linewidth=2)
    plt.plot(forecast.index, forecast.values, label='Ensemble Forecast', linestyle='--')
    plt.title('Ensemble Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Forecast plot saved to {save_path}")

# === MAIN SCRIPT ===

def main(ticker):
    ticker_upper = ticker.upper()
    ticker_lower = ticker.lower()

    BASE_DIR = r"C:\Users\GODIWOUR\Documents\Confidential\ALX_DS\stock-forecasting-strategy"

    forecast_paths = {
        'arima': os.path.join(BASE_DIR, 'results', 'arima', ticker_upper, f'{ticker_lower}_forecast.csv'),
        'lstm': os.path.join(BASE_DIR, 'results', 'lstm', ticker_upper, f'{ticker_lower}_forecast.csv'),
        'prophet': os.path.join(BASE_DIR, 'results', 'prophet', ticker_upper, f'{ticker_lower}_forecast.csv'),
        'xgboost': os.path.join(BASE_DIR, 'results', 'xgboost', ticker_upper, f'{ticker_lower}_forecast.csv')
    }

    actual_path = os.path.join(BASE_DIR, 'data', f'{ticker_upper}.csv')
    results_dir = os.path.join(BASE_DIR, 'results', 'ensemble', ticker_upper)
    os.makedirs(results_dir, exist_ok=True)

    print(f"📥 Loading forecasts for {ticker_upper}...")
    forecasts = {model: load_forecast(path)['Forecast'] for model, path in forecast_paths.items()}
    df_forecasts = pd.concat(forecasts.values(), axis=1)
    df_forecasts.columns = forecasts.keys()

    print("⚖️ Applying weighted ensemble strategy...")
    weights = {
        'arima': 0.1,
        'lstm': 0.3,
        'prophet': 0.1,
        'xgboost': 0.5
    }

    ensemble_forecast = sum(df_forecasts[model] * weights[model] for model in forecasts)
    ensemble_df = pd.DataFrame({
        'Date': df_forecasts.index,
        'Forecast': ensemble_forecast
    }).set_index('Date')

    value_col = f'{ticker_upper}.3'  # assumes this naming in CSVs
    actual = load_actual(actual_path, value_col=value_col)

    # Align actual and forecast to the same dates (last N days)
    common_idx = ensemble_df.index.intersection(actual.index)
    actual_aligned = actual.loc[common_idx]
    forecast_aligned = ensemble_df['Forecast'].loc[common_idx]

    # Drop any rows with NaN in either actual or forecast
    aligned_df = pd.DataFrame({'actual': actual_aligned, 'forecast': forecast_aligned}).dropna()
    actual_aligned = aligned_df['actual']
    forecast_aligned = aligned_df['forecast']

    print("✅ Evaluating ensemble model...")
    results = evaluate_forecast(actual_aligned.values, forecast_aligned.values, model_name=f"Ensemble_{ticker_upper}")
    print_evaluation(results)

    forecast_path = os.path.join(results_dir, f'{ticker_lower}_ensemble_forecast.csv')
    eval_path = os.path.join(results_dir, f'{ticker_lower}_ensemble_evaluation.txt')
    plot_path = os.path.join(results_dir, f'{ticker_lower}_ensemble_plot.png')

    save_forecast(ensemble_df, forecast_path)
    save_evaluation(results, eval_path)
    plot_ensemble(actual, ensemble_df['Forecast'], plot_path)

# === CLI SUPPORT ===

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Ensemble Forecast for a specific ticker.')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol, e.g., AAPL, MSFT, PFE')
    args = parser.parse_args()
    main(args.ticker)
