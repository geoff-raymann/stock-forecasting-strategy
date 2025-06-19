# app/executive_dashboard.py

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from glob import glob
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from evaluate_models import parse_evaluation_file  # You must have this util implemented

# === CONFIG ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
ALL_MODELS = ['arima', 'lstm', 'prophet', 'xgboost', 'ensemble_reverse_rsme']

# === HELPER FUNCTIONS ===
def get_available_tickers():
    tickers = set()
    for model in ALL_MODELS:
        model_dir = os.path.join(RESULTS_DIR, model)
        if os.path.isdir(model_dir):
            tickers.update(os.listdir(model_dir))
    return sorted(tickers)

def load_forecast(model, ticker):
    path = os.path.join(RESULTS_DIR, model, ticker, f'{ticker.lower()}_forecast.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if 'Date' not in df.columns or 'Forecast' not in df.columns:
        return None
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def load_evaluation(model, ticker):
    path = os.path.join(RESULTS_DIR, model, ticker, f'{ticker.lower()}_evaluation.txt')
    if not os.path.exists(path):
        return None
    return parse_evaluation_file(path)  # Should return a dict

def plot_forecast(ticker, forecasts):
    fig = plt.figure(figsize=(10, 5))
    for model, df in forecasts.items():
        plt.plot(df['Date'], df['Forecast'], label=model.upper())
    plt.title(f"Forecast Comparison for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig)

def display_metrics(ticker, evals):
    st.subheader(f"📊 Evaluation Metrics for {ticker}")
    cols = st.columns(len(evals))
    for idx, (model, metrics) in enumerate(evals.items()):
        with cols[idx]:
            st.metric("Model", model.upper())
            st.metric("MAE", f"{metrics['MAE']:.2f}")
            st.metric("MSE", f"{metrics['MSE']:.2f}")
            st.metric("RMSE", f"{metrics['RMSE']:.2f}")

def plot_comparison_chart(ticker, evals):
    chart_data = pd.DataFrame([
        {"Model": model.upper(), "Metric": "RMSE", "Value": metrics['RMSE']}
        for model, metrics in evals.items()
    ])
    fig = px.bar(chart_data, x='Model', y='Value', color='Model', text='Value',
                 title=f"RMSE Comparison - {ticker}", height=400)
    st.plotly_chart(fig)

# === STREAMLIT UI ===
st.set_page_config(page_title="📈 Executive Forecast Dashboard", layout="wide")
st.title("📊 Executive Dashboard: Stock Forecasting Comparison")

with st.sidebar:
    st.header("Filters")
    selected_tickers = st.multiselect("Choose Ticker(s)", get_available_tickers(), default=["AAPL"])
    selected_models = st.multiselect("Choose Model(s)", ALL_MODELS, default=['lstm', 'xgboost', 'ensemble_reverse_rsme'])

st.markdown("---")

for ticker in selected_tickers:
    st.header(f"📈 {ticker} Forecast Summary")
    forecasts = {}
    evaluations = {}

    for model in selected_models:
        df = load_forecast(model, ticker)
        if df is not None:
            forecasts[model] = df
        metrics = load_evaluation(model, ticker)
        if metrics:
            evaluations[model] = metrics

    if forecasts:
        plot_forecast(ticker, forecasts)
    if evaluations:
        display_metrics(ticker, evaluations)
        plot_comparison_chart(ticker, evaluations)

    st.markdown("---")

st.success("✅ Dashboard Auto-Loaded from Model Outputs")
