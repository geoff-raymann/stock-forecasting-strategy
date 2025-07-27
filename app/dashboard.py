import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
ENSEMBLE_DIR = os.path.join(RESULTS_DIR, 'ensemble_reverse_rsme')
MONTE_CARLO_DIR = os.path.join(RESULTS_DIR, 'monte_carlo')
SUMMARY_FILE = os.path.join(RESULTS_DIR, 'all_models_dashboard_summary.csv')
MC_SUMMARY_FILE = os.path.join(MONTE_CARLO_DIR, 'monte_carlo_summary.csv')

ALL_MODELS = ['LSTM', 'XGBoost', 'Ensemble']


def load_summary_data():
    if os.path.exists(SUMMARY_FILE):
        return pd.read_csv(SUMMARY_FILE)
    return pd.DataFrame()


def load_mc_summary_data():
    if os.path.exists(MC_SUMMARY_FILE):
        return pd.read_csv(MC_SUMMARY_FILE)
    return pd.DataFrame()


def load_forecast(ticker, model):
    path = os.path.join(RESULTS_DIR, model.lower(), ticker, f"{ticker.lower()}_forecast.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def plot_forecast(df, model):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(df['Date']), df['Forecast'], label='Forecast')
    ax.set_title(f"{model} Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    return fig


def plot_leaderboard(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df.sort_values('RMSE'), x='RMSE', y='Model', palette='Blues_d', ax=ax)
    ax.set_title("Model Leaderboard (Lower RMSE = Better)")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Model")
    return fig


def plot_monte_carlo(ticker):
    path = os.path.join(MONTE_CARLO_DIR, ticker, f"{ticker.lower()}_simulations.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in df.columns[:100]:
        ax.plot(df.index, df[col], color='skyblue', alpha=0.05)
    ax.set_title(f"Monte Carlo Simulation - {ticker} (First 100 Paths)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Simulated Price")
    return fig


def plot_mc_distribution(ticker):
    path = os.path.join(MONTE_CARLO_DIR, ticker, f"{ticker.lower()}_simulations.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    final_prices = df.iloc[-1, :]
    mean = final_prices.mean()
    p5 = final_prices.quantile(0.05)
    p95 = final_prices.quantile(0.95)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(final_prices, bins=50, kde=True, ax=ax, color='skyblue')
    ax.axvline(mean, color='blue', linestyle='--', label=f"Mean: {mean:.2f}")
    ax.axvline(p5, color='red', linestyle='--', label=f"5th %: {p5:.2f}")
    ax.axvline(p95, color='green', linestyle='--', label=f"95th %: {p95:.2f}")
    ax.set_title(f"Monte Carlo Final Price Distribution - {ticker}")
    ax.set_xlabel("Simulated Final Price")
    ax.legend()
    return fig


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


def convert_fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("📈 Stock Forecast Dashboard")

summary_df = load_summary_data()
mc_summary_df = load_mc_summary_data()

if summary_df.empty:
    st.warning("No forecast summary available.")
else:
    tickers = summary_df['Ticker'].unique().tolist()
    selected_ticker = st.selectbox("Select a Ticker:", tickers)

    st.subheader("📊 Forecast Summary")
    ticker_summary = summary_df[summary_df['Ticker'] == selected_ticker]
    st.dataframe(ticker_summary[['Model', 'MAE', 'MSE', 'RMSE', 'Timestamp']].sort_values('RMSE'))

    csv = convert_df(ticker_summary)
    st.download_button("⬇️ Download Forecast Summary CSV", csv, f"{selected_ticker}_forecast_summary.csv", "text/csv")

    st.markdown("---")
    st.subheader("🏆 Model Leaderboard")
    leaderboard_fig = plot_leaderboard(ticker_summary)
    st.pyplot(leaderboard_fig)

    st.subheader("📉 Forecast Charts")
    forecast_cols = st.columns(len(ALL_MODELS))
    for i, model in enumerate(ALL_MODELS):
        with forecast_cols[i]:
            forecast_df = load_forecast(selected_ticker, model)
            if forecast_df is not None:
                fig = plot_forecast(forecast_df, model)
                st.pyplot(fig)
                st.download_button("⬇️ Download Chart", convert_fig_to_image(fig), f"{selected_ticker}_{model}_forecast.png", "image/png")

    st.markdown("---")
    st.subheader("🎲 Monte Carlo Simulation")
    mc_fig = plot_monte_carlo(selected_ticker)
    if mc_fig:
        st.pyplot(mc_fig)
        st.download_button("⬇️ Download Monte Carlo Chart", convert_fig_to_image(mc_fig), f"{selected_ticker}_monte_carlo.png", "image/png")
    else:
        st.warning("Monte Carlo data not available.")

    if not mc_summary_df.empty and selected_ticker in mc_summary_df['Ticker'].values:
        st.markdown("#### Monte Carlo Summary Statistics")
        mc_stats = mc_summary_df[mc_summary_df['Ticker'] == selected_ticker]
        st.dataframe(mc_stats)
        csv_mc = convert_df(mc_stats)
        st.download_button("⬇️ Download Monte Carlo Summary CSV", csv_mc, f"{selected_ticker}_monte_carlo_summary.csv", "text/csv")

        st.markdown("#### Distribution of Final Simulated Prices")
        mc_dist_fig = plot_mc_distribution(selected_ticker)
        st.pyplot(mc_dist_fig)
        st.download_button("⬇️ Download Distribution Chart", convert_fig_to_image(mc_dist_fig), f"{selected_ticker}_mc_distribution.png", "image/png")
