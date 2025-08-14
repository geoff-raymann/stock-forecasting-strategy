# src/ensemble_model.py

import os
import argparse
import pandas as pd
from datetime import datetime
from evaluate_models import evaluate_forecast, print_evaluation
import matplotlib.pyplot as plt

# === CONFIG ===
DEFAULT_TICKER = 'AAPL'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
ENSEMBLE_DIR = os.path.join(RESULTS_DIR, 'ensemble_reverse_rsme')
DATA_DIR = os.path.join(BASE_DIR, 'data')
DASHBOARD_SUMMARY_PATH = os.path.join(ENSEMBLE_DIR, 'ensemble_dashboard_summary.csv')

# === DYNAMIC WEIGHTS (adjust as needed) ===
DYNAMIC_WEIGHTS = {
    'AAPL': {'LSTM': 0.467, 'XGBoost': 0.533},
    'MSFT': {'LSTM': 0.758, 'XGBoost': 0.242},
    'PFE' : {'LSTM': 0.630, 'XGBoost': 0.370},
    'JNJ' : {'LSTM': 0.490, 'XGBoost': 0.510}
}

# === FUNCTIONS ===

def load_forecast(ticker, model):
    path = os.path.join(RESULTS_DIR, model.lower(), ticker, f'{ticker.lower()}_forecast.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Forecast file not found for {model}_{ticker} at {path}")
    df = pd.read_csv(path)
    return df['Forecast'].values, pd.to_datetime(df['Date'])

def ensemble_forecast(ticker):
    print(f"\n📥 Loading model forecasts for {ticker}...")

    if ticker not in DYNAMIC_WEIGHTS:
        raise ValueError(f"❌ No dynamic weights defined for '{ticker}' in DYNAMIC_WEIGHTS dictionary.")

    lstm_preds, dates = load_forecast(ticker, 'LSTM')
    xgb_preds, _ = load_forecast(ticker, 'XGBoost')

    weights = DYNAMIC_WEIGHTS[ticker]
    ensemble = (weights['LSTM'] * lstm_preds) + (weights['XGBoost'] * xgb_preds)
    return ensemble, dates

def load_actuals(ticker):
    path = os.path.join(DATA_DIR, f'{ticker}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Data file for {ticker} not found at {path}")
    
    df = pd.read_csv(path, header=1)
    df = df[df['Ticker'] != 'Date']
    df['Ticker'] = pd.to_datetime(df['Ticker'])
    df.set_index('Ticker', inplace=True)
    df = df.asfreq('B')

    close_col = f'{ticker}.3'
    if close_col not in df.columns:
        raise KeyError(f"❌ Expected column '{close_col}' not found in {ticker}.csv")

    df['Close'] = pd.to_numeric(df[close_col], errors='coerce').interpolate()
    return df['Close'].iloc[-30:].values

def save_forecast(dates, forecast, ticker):
    # Load actuals for the forecast period
    actual = load_actuals(ticker)
    # Ensure lengths match (if not, pad with NaN or slice)
    if len(actual) != len(forecast):
        # Try to align by slicing to the shortest length
        min_len = min(len(actual), len(forecast))
        actual = actual[-min_len:]
        forecast = forecast[-min_len:]
        dates = dates[-min_len:]
    ticker_dir = os.path.join(ENSEMBLE_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    path = os.path.join(ticker_dir, f'{ticker.lower()}_forecast.csv')
    pd.DataFrame({'Date': dates, 'Forecast': forecast, 'Actual': actual}).to_csv(path, index=False)
    print(f"📄 Forecast values saved to {path}")

def save_evaluation(results, ticker):
    ticker_dir = os.path.join(ENSEMBLE_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    path = os.path.join(ticker_dir, f'{ticker.lower()}_evaluation.txt')

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n📊 Evaluation for Ensemble\n")
        f.write(f"{'-'*30}\n")
        for k, v in results.items():
            if isinstance(v, (float, int)):
                f.write(f"{k.upper():<6}: {v:.4f}\n")
            else:
                f.write(f"{k.upper():<6}: {v}\n")
        f.write(f"\nSaved on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"📄 Evaluation results saved to {path}")

    summary_row = {
        'Ticker': ticker,
        'Model': 'Ensemble',
        'MAE': results.get('mae'),
        'MSE': results.get('mse'),
        'RMSE': results.get('rmse'),
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    if os.path.exists(DASHBOARD_SUMMARY_PATH):
        df = pd.read_csv(DASHBOARD_SUMMARY_PATH)
        df = df[df['Ticker'] != ticker]
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        df = pd.DataFrame([summary_row])

    df.to_csv(DASHBOARD_SUMMARY_PATH, index=False)
    print(f"📊 Dashboard summary updated at {DASHBOARD_SUMMARY_PATH}")

def save_forecast_plot(dates, actual, forecast, ticker):
    plt.figure(figsize=(10, 4))
    plt.plot(dates, actual, label='Actual', marker='o')
    plt.plot(dates, forecast, label='Ensemble Forecast', marker='x')
    plt.title(f"{ticker} - Actual vs Ensemble Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    ticker_dir = os.path.join(ENSEMBLE_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    path = os.path.join(ticker_dir, f"{ticker.lower()}_forecast_plot.png")
    plt.savefig(path)
    plt.close()
    print(f"📊 Forecast plot saved to {path}")

# === MAIN EXECUTION ===
def main(ticker=DEFAULT_TICKER):
    forecast, dates = ensemble_forecast(ticker)
    actual = load_actuals(ticker)
    results = evaluate_forecast(actual, forecast, model_name=f'Ensemble_{ticker}')
    print_evaluation(results)
    save_forecast(dates, forecast, ticker)
    save_forecast_plot(dates, actual, forecast, ticker)  # <-- Add this line
    save_evaluation(results, ticker)

# === CLI ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ensemble forecast combining LSTM + XGBoost using RMSE-based weights.')
    parser.add_argument('--ticker', type=str, default=DEFAULT_TICKER, help='Ticker symbol (e.g., AAPL, MSFT, etc.)')
    args = parser.parse_args()
    main(args.ticker)
