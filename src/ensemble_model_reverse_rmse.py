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
ENSEMBLE_DIR = os.path.join(RESULTS_DIR, 'ensemble')
os.makedirs(ENSEMBLE_DIR, exist_ok=True)

# === DYNAMIC WEIGHTS (based on inverse RMSE) ===
DYNAMIC_WEIGHTS = {
    'AAPL': {'LSTM': 0.467, 'XGBoost': 0.533},
    'MSFT': {'LSTM': 0.758, 'XGBoost': 0.242},
    'PFE' : {'LSTM': 0.630, 'XGBoost': 0.370}
}

# === FUNCTIONS ===
def load_forecast(ticker, model):
    forecast_path = os.path.join(RESULTS_DIR, model.lower(), ticker, f'{ticker.lower()}_forecast.csv')
    df = pd.read_csv(forecast_path)
    return df['Forecast'].values, pd.to_datetime(df['Date'])

def ensemble_forecast(ticker):
    print(f"📥 Loading model forecasts for {ticker}...")
    lstm_preds, dates = load_forecast(ticker, 'LSTM')
    xgb_preds, _ = load_forecast(ticker, 'XGBoost')

    weights = DYNAMIC_WEIGHTS[ticker]
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
    df['Close'] = pd.to_numeric(df[f'{ticker}.3'], errors='coerce')
    df['Close'] = df['Close'].interpolate(method='linear')
    return df['Close'].iloc[-30:].values
def save_forecast(dates, forecast, ticker):
    path = os.path.join(ENSEMBLE_DIR, f'{ticker.lower()}_forecast.csv')
    df = pd.DataFrame({ 'Date': dates, 'Forecast': forecast })
    df.to_csv(path, index=False)
    print(f"📄 Forecast saved to {path}")


def save_evaluation(results, ticker):
    path = os.path.join(ENSEMBLE_DIR, f'{ticker.lower()}_evaluation.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n📊 Evaluation for Ensemble\n")
        f.write(f"{'-'*30}\n")
        for k, v in results.items():
            if isinstance(v, (float, int)):
                f.write(f"{k.upper():<6}: {v:.4f}\n")
            else:
                f.write(f"{k.upper():<6}: {v}\n")
        f.write(f"\nSaved on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"📄 Evaluation saved to {path}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default=DEFAULT_TICKER)
    args = parser.parse_args()
    main(args.ticker)
