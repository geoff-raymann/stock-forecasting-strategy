# src/ensemble_model.py

import os
import argparse
import pandas as pd
from datetime import datetime

from evaluate_models import evaluate_forecast, print_evaluation

# === CONFIG ===
DEFAULT_TICKER = 'AAPL'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
ENSEMBLE_DIR = os.path.join(RESULTS_DIR, 'ensemble_reverse_rsme')
DASHBOARD_SUMMARY_PATH = os.path.join(ENSEMBLE_DIR, 'ensemble_dashboard_summary.csv')

# === DYNAMIC WEIGHTS (based on inverse RMSE) ===
DYNAMIC_WEIGHTS = {
    'AAPL': {'LSTM': 0.467, 'XGBoost': 0.533},
    'MSFT': {'LSTM': 0.758, 'XGBoost': 0.242},
    'PFE' : {'LSTM': 0.630, 'XGBoost': 0.370},
    'JNJ' : {'LSTM': 0.490, 'XGBoost': 0.510}
}

# === FUNCTIONS ===
def load_forecast(ticker, model):
    forecast_path = os.path.join(RESULTS_DIR, model.lower(), ticker, f'{ticker.lower()}_forecast.csv')
    df = pd.read_csv(forecast_path)
    return df['Forecast'].values, pd.to_datetime(df['Date'])

def ensemble_forecast(ticker):
    print(f"\n📥 Loading model forecasts for {ticker}...")
    lstm_preds, dates = load_forecast(ticker, 'LSTM')
    xgb_preds, _ = load_forecast(ticker, 'XGBoost')

    weights = DYNAMIC_WEIGHTS.get(ticker)
    if not weights:
        raise ValueError(f"❌ No dynamic weights found for ticker '{ticker}'. Please add them to DYNAMIC_WEIGHTS.")

    ensemble = (weights['LSTM'] * lstm_preds) + (weights['XGBoost'] * xgb_preds)
    return ensemble, dates

def load_actuals(ticker):
    path = os.path.join(BASE_DIR, 'data', f'{ticker}.csv')
    df = pd.read_csv(path, header=1)
    df = df[df['Ticker'] != 'Date']
    df['Ticker'] = pd.to_datetime(df['Ticker'])
    df.set_index('Ticker', inplace=True)
    df = df.asfreq('B')
    df['Close'] = pd.to_numeric(df[f'{ticker}.3'], errors='coerce')
    df['Close'] = df['Close'].interpolate(method='linear')
    return df['Close'].iloc[-30:].values

def save_forecast(dates, forecast, ticker):
    ticker_dir = os.path.join(ENSEMBLE_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    path = os.path.join(ticker_dir, f'{ticker.lower()}_forecast.csv')
    df = pd.DataFrame({'Date': dates, 'Forecast': forecast})
    df.to_csv(path, index=False)
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

    # Append or update dashboard summary CSV
    summary_row = {
        'Ticker': ticker,
        'Model': 'Ensemble',
        'MAE': results.get('mae'),
        'MSE': results.get('mse'),
        'RMSE': results.get('rmse'),
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    if os.path.exists(DASHBOARD_SUMMARY_PATH):
        summary_df = pd.read_csv(DASHBOARD_SUMMARY_PATH)
        summary_df = summary_df[summary_df['Ticker'] != ticker]  # Remove previous entry
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(DASHBOARD_SUMMARY_PATH, index=False)
    print(f"📊 Dashboard summary updated at {DASHBOARD_SUMMARY_PATH}")

# === MAIN ===
def main(ticker=DEFAULT_TICKER):
    forecast, dates = ensemble_forecast(ticker)
    actual = load_actuals(ticker)
    results = evaluate_forecast(actual, forecast, model_name=f'Ensemble_{ticker}')
    print_evaluation(results)
    save_forecast(dates, forecast, ticker)
    save_evaluation(results, ticker)

# === CLI ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ensemble forecast combining LSTM + XGBoost using RMSE-based weights.')
    parser.add_argument('--ticker', type=str, default=DEFAULT_TICKER, help='Ticker symbol (e.g., AAPL, MSFT, PFE, JNJ)')
    args = parser.parse_args()
    main(args.ticker)
