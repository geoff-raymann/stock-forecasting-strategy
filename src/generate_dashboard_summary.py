# src/generate_dashboard_summary.py

import os
import re
import pandas as pd

ENSEMBLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'ensemble_reverse_rsme'))
output_csv = os.path.join(ENSEMBLE_DIR, 'ensemble_dashboard_summary.csv')

summary_data = []

for fname in os.listdir(ENSEMBLE_DIR):
    if fname.endswith('_evaluation.txt'):
        ticker = fname.split('_')[0].upper()
        path = os.path.join(ENSEMBLE_DIR, fname)
        with open(path, 'r') as f:
            content = f.read()
            mae = float(re.search(r'MAE\s+:\s+([\d.]+)', content).group(1))
            mse = float(re.search(r'MSE\s+:\s+([\d.]+)', content).group(1))
            rmse = float(re.search(r'RMSE\s+:\s+([\d.]+)', content).group(1))
            summary_data.append({
                'Ticker': ticker,
                'Model': 'Ensemble',
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse
            })

# Save summary
df = pd.DataFrame(summary_data)
if not df.empty and 'RMSE' in df.columns:
    df.sort_values('RMSE', inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"📄 Summary dashboard saved to {output_csv}")
else:
    print("⚠️ No summary data found or 'RMSE' column missing. No CSV file was created.")
