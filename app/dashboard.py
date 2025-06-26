# app/dashboard.py

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from glob import glob
import base64
from io import BytesIO
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from evaluate_models import parse_evaluation_file  # You must have this util implemented

# === CONFIG ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')
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
    return parse_evaluation_file(path)

def load_actuals(ticker):
    path = os.path.join(DATA_DIR, f'{ticker}.csv')
    df = pd.read_csv(path, header=1)
    df = df[df['Ticker'] != 'Date']
    df['Ticker'] = pd.to_datetime(df['Ticker'])
    df = df.set_index('Ticker').asfreq('B').reset_index()
    df['Close'] = pd.to_numeric(df[f'{ticker}.3'], errors='coerce')
    df['Close'] = df['Close'].interpolate(method='linear')
    return df[['Ticker', 'Close']].rename(columns={'Ticker': 'Date'})

def plot_forecast(ticker, forecasts, actuals):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actuals['Date'], y=actuals['Close'], mode='lines', name='Actual'))
    for model, df in forecasts.items():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Forecast'], mode='lines', name=model.upper()))
    fig.update_layout(title=f"Forecast Comparison for {ticker}", xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)

def display_metrics(ticker, evals):
    st.subheader(f"📊 Evaluation Metrics for {ticker}")
    df = pd.DataFrame([{
        'Model': model.upper(),
        'MAE': metrics['MAE'],
        'MSE': metrics['MSE'],
        'RMSE': metrics['RMSE']
    } for model, metrics in evals.items()])
    st.dataframe(df.style.format({"MAE": "{:.2f}", "MSE": "{:.2f}", "RMSE": "{:.2f}"}))
    return df

def plot_comparison_chart(ticker, evals):
    chart_data = pd.DataFrame([
        {"Model": model.upper(), "Metric": "RMSE", "Value": metrics['RMSE']}
        for model, metrics in evals.items()
    ])
    fig = px.bar(chart_data, x='Model', y='Value', color='Model', text='Value',
                 title=f"RMSE Comparison - {ticker}", height=400)
    st.plotly_chart(fig)

def generate_download_links(df, filename_prefix):
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    excel_data = buffer.getvalue()
    st.download_button("⬇️ Download Excel", data=excel_data, file_name=f"{filename_prefix}.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    pdf_html = df.to_html(index=False)
    b64 = base64.b64encode(pdf_html.encode()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename_prefix}.html">⬇️ Download HTML Summary</a>'
    st.markdown(href, unsafe_allow_html=True)

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

    actuals = load_actuals(ticker)
    if forecasts:
        plot_forecast(ticker, forecasts, actuals)
    if evaluations:
        summary_df = display_metrics(ticker, evaluations)
        plot_comparison_chart(ticker, evaluations)
        generate_download_links(summary_df, filename_prefix=f"{ticker}_model_metrics")

    st.markdown("---")

st.success("✅ Dashboard Auto-Loaded from Model Outputs")
