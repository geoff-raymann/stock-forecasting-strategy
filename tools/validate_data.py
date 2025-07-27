# tools/validate_data.py

import os
import pandas as pd

# === CONFIG ===
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
TICKERS = ['AAPL', 'MSFT', 'PFE', 'JNJ']  # Keep in sync with data_loader.py

REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

def validate_file(ticker):
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")

    if not os.path.exists(file_path):
        return f"❌ {ticker}: File not found."

    try:
        df = pd.read_csv(file_path)

        if df.empty:
            return f"❌ {ticker}: File is empty."

        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            return f"❌ {ticker}: Missing columns: {missing_cols}"

        if df['Close'].isnull().all():
            return f"❌ {ticker}: All Close prices are NaN."

        return f"✅ {ticker}: File OK."

    except Exception as e:
        return f"❌ {ticker}: Error reading file — {e}"

def main():
    print(f"\n🔍 Validating ticker files in '{DATA_DIR}'\n")
    for ticker in TICKERS:
        print(validate_file(ticker))

if __name__ == '__main__':
    main()
