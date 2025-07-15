# app/dashboard.py
import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from evaluate_models import parse_evaluation_file

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ALL_MODELS = ['arima', 'lstm', 'prophet', 'xgboost', 'ensemble_reverse_rsme']

def get_available_tickers():
    tickers = set()
    for model in ALL_MODELS:
        model_dir = os.path.join(RESULTS_DIR, model)
        if os.path.isdir(model_dir):
            tickers.update(os.listdir(model_dir))
    return sorted(tickers)

def load_forecast(model, ticker):
    path = os.path.join(RESULTS_DIR, model, ticker, f'{ticker.lower()}_forecast.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

def load_actuals(ticker):
    path = os.path.join(DATA_DIR, f'{ticker}.csv')
    df = pd.read_csv(path, header=1)
    df = df[df['Ticker'] != 'Date']
    df['Ticker'] = pd.to_datetime(df['Ticker'])
    df = df.set_index('Ticker').asfreq('B').reset_index()
    df['Close'] = pd.to_numeric(df[f'{ticker}.3'], errors='coerce').interpolate()
    return df[['Ticker', 'Close']].rename(columns={'Ticker': 'Date'})

def load_evaluation(model, ticker):
    path = os.path.join(RESULTS_DIR, model, ticker, f'{ticker.lower()}_evaluation.txt')
    if os.path.exists(path):
        return parse_evaluation_file(path)
    return None

def plot_forecast(ticker, forecasts, actuals):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actuals['Date'], y=actuals['Close'], name="Actual"))
    for model, df in forecasts.items():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Forecast'], name=model.upper()))
    fig.update_layout(title=f"{ticker} Forecast vs Actual", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

def display_metrics(ticker, evals):
    st.subheader(f"📊 Metrics for {ticker}")
    df = pd.DataFrame([{
        'Model': model.upper(),
        'MAE': metrics['MAE'],
        'MSE': metrics['MSE'],
        'RMSE': metrics['RMSE']
    } for model, metrics in evals.items()])
    st.dataframe(df.style.format({"MAE": "{:.2f}", "MSE": "{:.2f}", "RMSE": "{:.2f}"}))
    return df

def generate_download_links(df, filename_prefix):
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    st.download_button("⬇️ Download Excel", data=buffer.getvalue(),
                       file_name=f"{filename_prefix}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    html = df.to_html(index=False)
    b64 = base64.b64encode(html.encode()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename_prefix}.html">⬇️ Download HTML Summary</a>'
    st.markdown(href, unsafe_allow_html=True)

# Streamlit layout
st.set_page_config(page_title="Executive Dashboard", layout="wide")
st.title("📊 Stock Forecasting Executive Dashboard")

with st.sidebar:
    st.header("Filters")
    tickers = get_available_tickers()
    selected_tickers = st.multiselect("Ticker(s)", tickers, default=tickers[:1])
    selected_models = st.multiselect("Models", ALL_MODELS, default=ALL_MODELS)

st.markdown("---")

for ticker in selected_tickers:
    st.header(f"{ticker} Forecast")
    forecasts = {model: load_forecast(model, ticker) for model in selected_models if load_forecast(model, ticker) is not None}
    actuals = load_actuals(ticker)
    evals = {model: load_evaluation(model, ticker) for model in selected_models if load_evaluation(model, ticker)}

    if forecasts:
        plot_forecast(ticker, forecasts, actuals)
    if evals:
        df = display_metrics(ticker, evals)
        generate_download_links(df, f"{ticker}_metrics")
    st.markdown("---")
