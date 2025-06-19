# tools/validate_forecasts.py

import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS = ['arima', 'lstm', 'prophet', 'xgboost', 'ensemble', 'ensemble_reverse_rsme']

def validate_forecast_file(path, model, ticker):
    try:
        df = pd.read_csv(path)
        required_cols = ['Date', 'Forecast']
        missing = [col for col in required_cols if col not in df.columns]

        if df.empty:
            return f"❌ {model.upper()} - {ticker}: File is empty."
        elif missing:
            return f"❌ {model.upper()} - {ticker}: Missing columns {missing}"
        return f"✅ {model.upper()} - {ticker}: OK"
    except Exception as e:
        return f"❌ {model.upper()} - {ticker}: Error reading file - {str(e)}"

def main():
    print("🔍 Validating forecast output files...\n")
    for model in MODELS:
        model_dir = os.path.join(RESULTS_DIR, model)
        if not os.path.exists(model_dir):
            print(f"⚠️ Skipping missing directory: {model_dir}")
            continue

        for ticker_folder in os.listdir(model_dir):
            forecast_file = os.path.join(model_dir, ticker_folder, f'{ticker_folder.lower()}_forecast.csv')
            if os.path.exists(forecast_file):
                result = validate_forecast_file(forecast_file, model, ticker_folder)
                print(result)

if __name__ == '__main__':
    main()
