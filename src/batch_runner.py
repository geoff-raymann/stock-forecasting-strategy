import os
import subprocess
import sys
import pandas as pd
from datetime import datetime

# === CONFIG ===
TICKERS = ['AAPL', 'MSFT', 'PFE', 'JNJ']
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
MC_DIR = os.path.join(RESULTS_DIR, 'monte_carlo')
ENSEMBLE_DIR = os.path.join(RESULTS_DIR, 'ensemble_reverse_rsme')
DASHBOARD_CSV = os.path.join(ENSEMBLE_DIR, 'ensemble_dashboard_summary.csv')

def run_script(description, command):
    print(f"\n🚀 Running: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ Finished: {command}")
    except subprocess.CalledProcessError:
        print(f"❌ Command failed: {command}")

def validate_data():
    print("\n🔍 Validating data before forecast run...")
    run_script("Validating data", f"python {os.path.join(SRC_DIR, '../tools/validate_data.py')}")

def run_models(ticker):
    print(f"\n📈 Running models for {ticker}...")
    run_script("ARIMA", f"python {os.path.join(SRC_DIR, 'arima_model.py')} --ticker {ticker}")
    run_script("LSTM", f"python {os.path.join(SRC_DIR, 'lstm_model.py')} --ticker {ticker}")
    run_script("Prophet", f"python {os.path.join(SRC_DIR, 'facebook_prophet_model.py')} --ticker {ticker}")
    run_script("XGBoost", f"python {os.path.join(SRC_DIR, 'xgboost_model.py')} --ticker {ticker}")
    run_script("Ensemble", f"python {os.path.join(SRC_DIR, 'ensemble_model_reverse_rmse.py')} --ticker {ticker}")

def run_monte_carlo(ticker):
    print("\n🎲 Running Monte Carlo scenario simulation...")
    run_script("Monte Carlo", f"python {os.path.join(SRC_DIR, 'monte_carlo_simulation.py')} --ticker {ticker}")

def generate_dashboard():
    print("\n🗂️ Generating dashboard summary CSV...")
    run_script("Dashboard Summary", f"python {os.path.join(SRC_DIR, 'generate_dashboard_summary.py')}")

if __name__ == '__main__':
    validate_data()
    for ticker in TICKERS:
        run_models(ticker)
        run_monte_carlo(ticker)
    generate_dashboard()