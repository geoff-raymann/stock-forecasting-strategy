# src/generate_monte_carlo_summary.py

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MC_DIR = os.path.join(BASE_DIR, 'results', 'monte_carlo')
OUTPUT_PATH = os.path.join(MC_DIR, 'monte_carlo_summary.csv')

summary = []

for ticker_folder in os.listdir(MC_DIR):
    ticker_path = os.path.join(MC_DIR, ticker_folder)
    if not os.path.isdir(ticker_path):
        continue

    csv_file = os.path.join(ticker_path, f'{ticker_folder.lower()}_simulations.csv')
    if not os.path.exists(csv_file):
        print(f"⚠️ Missing simulation file for {ticker_folder}")
        continue

    try:
        df = pd.read_csv(csv_file)
        final_prices = df.tail(1).values.flatten()
        mean_price = np.mean(final_prices)
        std_dev = np.std(final_prices)
        summary.append({
            'Ticker': ticker_folder,
            'MeanFinalPrice': round(mean_price, 2),
            'Volatility': round(std_dev, 2),
            'Simulations': df.shape[1],
            'Days': df.shape[0]
        })
    except Exception as e:
        print(f"❌ Error processing {csv_file}: {e}")

# Save
summary_df = pd.DataFrame(summary)
if not summary_df.empty:
    summary_df.sort_values(by='Volatility', ascending=False, inplace=True)
    summary_df.to_csv(OUTPUT_PATH, index=False)
    print(f"📊 Monte Carlo summary saved to {OUTPUT_PATH}")
else:
    print("⚠️ No summary data found. No file was created.")
