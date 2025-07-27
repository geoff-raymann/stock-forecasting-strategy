# src/generate_dashboard_summary.py

import os
import re
import pandas as pd
from datetime import datetime

BASE_RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
OUTPUT_CSV = os.path.join(BASE_RESULTS_DIR, 'all_models_dashboard_summary.csv')

# All models to check
MODELS = ['arima', 'lstm', 'prophet', 'xgboost', 'ensemble_reverse_rsme']
SUMMARY = []

def extract_metrics(content):
    try:
        mae = re.search(r'MAE\s*:\s*([\d.]+)', content)
        mse = re.search(r'MSE\s*:\s*([\d.]+)', content)
        rmse = re.search(r'RMSE\s*:\s*([\d.]+)', content)
        timestamp = re.search(r'Saved on\s+([^\n]+)', content)

        return {
            'MAE': float(mae.group(1)) if mae else None,
            'MSE': float(mse.group(1)) if mse else None,
            'RMSE': float(rmse.group(1)) if rmse else None,
            'Timestamp': timestamp.group(1).strip() if timestamp else ''
        }
    except Exception as e:
        print(f"❌ Failed to extract metrics: {e}")
        return None

for model in MODELS:
    model_dir = os.path.join(BASE_RESULTS_DIR, model)
    if not os.path.isdir(model_dir):
        continue

    for ticker in os.listdir(model_dir):
        ticker_path = os.path.join(model_dir, ticker)
        if not os.path.isdir(ticker_path):
            continue

        eval_path = os.path.join(ticker_path, f"{ticker.lower()}_evaluation.txt")
        if os.path.exists(eval_path):
            with open(eval_path, 'r', encoding='utf-8') as f:
                content = f.read()
                metrics = extract_metrics(content)
                if metrics:
                    model_name = model.replace('_reverse_rsme', '').upper()
                    SUMMARY.append({
                        'Ticker': ticker.upper(),
                        'Model': model_name,
                        **metrics
                    })

# Save to DataFrame
df = pd.DataFrame(SUMMARY)

if not df.empty:
    df = df[['Ticker', 'Model', 'MAE', 'MSE', 'RMSE', 'Timestamp']]  # enforce column order
    df.sort_values(by=['Ticker', 'Model'], inplace=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"📊 All-model summary saved to {OUTPUT_CSV}")
else:
    print("⚠️ No evaluation data found. No summary file written.")
