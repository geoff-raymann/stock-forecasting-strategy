# src/monte_carlo_simulation.py

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# === CONFIG ===
DEFAULT_TICKER = 'AAPL'
DEFAULT_DATE_COL = 'Ticker'
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'monte_carlo'))

def load_data(ticker, date_col, value_col):
    file_path = os.path.join(DEFAULT_DATA_DIR, f'{ticker}.csv')
    df = pd.read_csv(file_path, header=1)

    # Drop 2nd header row if present
    df = df[df[date_col] != 'Date'].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

    df = df[[date_col, value_col]].dropna()
    df.columns = ['Date', 'Close']
    df = df.set_index('Date').asfreq('B')
    df['Close'].interpolate(method='linear', inplace=True)
    return df

def simulate_monte_carlo(df, n_simulations=1000, n_days=30):
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()
    S0 = df['Close'].iloc[-1]

    simulations = np.zeros((n_days, n_simulations))
    for i in range(n_simulations):
        daily_returns = np.random.normal(mu, sigma, n_days)
        price_series = S0 * np.exp(np.cumsum(daily_returns))
        simulations[:, i] = price_series
    return simulations

def plot_simulation(simulations, ticker, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(simulations, color='skyblue', alpha=0.05)
    plt.title(f"Monte Carlo Simulation ({ticker}) - 30 Business Days")
    plt.xlabel("Days")
    plt.ylabel("Simulated Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📸 Simulation plot saved to {save_path}")

def save_simulation_data(simulations, ticker, save_path):
    df = pd.DataFrame(simulations)
    df.to_csv(save_path, index=False)
    print(f"📄 Simulation data saved to {save_path}")

def main(ticker=DEFAULT_TICKER, date_col=None, value_col=None, simulations=1000, days=30):
    if not date_col:
        date_col = DEFAULT_DATE_COL
    if not value_col:
        value_col = f'{ticker}.3'

    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, ticker)
    os.makedirs(output_dir, exist_ok=True)

    print(f"📥 Loading data for {ticker}...")
    df = load_data(ticker, date_col, value_col)

    print("🎲 Running Monte Carlo simulation...")
    simulations = simulate_monte_carlo(df, n_simulations=simulations, n_days=days)

    print("💾 Saving results...")
    plot_path = os.path.join(output_dir, f'{ticker.lower()}_monte_carlo_plot.png')
    csv_path = os.path.join(output_dir, f'{ticker.lower()}_simulations.csv')

    plot_simulation(simulations, ticker, plot_path)
    save_simulation_data(simulations, ticker, csv_path)

    print(f"✅ Monte Carlo simulation completed for {ticker}")

# === CLI SUPPORT ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Monte Carlo Simulation for a given stock ticker.')
    parser.add_argument('--ticker', type=str, default=DEFAULT_TICKER, help='Ticker symbol (e.g., AAPL)')
    parser.add_argument('--date_col', type=str, help='Column name with dates')
    parser.add_argument('--value_col', type=str, help='Column name with closing prices')
    parser.add_argument('--simulations', type=int, default=1000, help='Number of simulations')
    parser.add_argument('--days', type=int, default=30, help='Number of days to simulate')
    args = parser.parse_args()
    main(ticker=args.ticker, date_col=args.date_col, value_col=args.value_col, simulations=args.simulations, days=args.days)