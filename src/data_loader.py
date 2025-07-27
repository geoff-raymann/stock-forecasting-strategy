# src/data_loader.py

import yfinance as yf
import os
import pandas as pd

# === CONFIG ===
TICKERS = ['AAPL', 'MSFT', 'PFE', 'JNJ']  # Add more if needed
START_DATE = '2019-01-01'
END_DATE = None  # Default = today
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

def fetch_ticker_data(ticker: str, start=START_DATE, end=END_DATE):
    print(f"📥 Fetching {ticker} data from Yahoo Finance...")

    df = yf.download(ticker, start=start, end=end, auto_adjust=False)

    if df.empty:
        print(f"⚠️ No data returned for {ticker}")
        return

    # Save exactly as-is — let model scripts reshape/wrangle as needed
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(out_path)
    print(f"✅ Saved {ticker} to {out_path}")

def main():
    for ticker in TICKERS:
        fetch_ticker_data(ticker)

if __name__ == '__main__':
    main()
