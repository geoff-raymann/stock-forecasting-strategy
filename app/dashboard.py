import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
ENSEMBLE_DIR = os.path.join(RESULTS_DIR, 'ensemble_reverse_rsme')
MONTE_CARLO_DIR = os.path.join(RESULTS_DIR, 'monte_carlo')
SUMMARY_FILE = os.path.join(RESULTS_DIR, 'all_models_dashboard_summary.csv')
MC_SUMMARY_FILE = os.path.join(MONTE_CARLO_DIR, 'monte_carlo_summary.csv')
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

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
    ticker_folder = ticker.upper()
    ticker_file = f"{ticker.lower()}_forecast.csv"
    if model == "Ensemble":
        path = os.path.join(ENSEMBLE_DIR, ticker_folder, ticker_file)
    else:
        path = os.path.join(RESULTS_DIR, model.lower(), ticker_folder, ticker_file)
    print(f"Trying to load forecast from: {path}")
    if os.path.exists(path):
        return pd.read_csv(path)
    print("File not found!")
    return None

def load_actuals(ticker):
    data_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # If 'Date' is not a column, try to get it from the index
        if 'Date' not in df.columns:
            df = df.reset_index()
            if 'Date' not in df.columns and 'index' in df.columns:
                df = df.rename(columns={'index': 'Date'})
        # Try common column names for price
        for col in ['Close', 'Adj Close', 'close', 'adjclose']:
            if col in df.columns and 'Date' in df.columns:
                return df[['Date', col]].rename(columns={col: 'Actual'})
    return None

def plot_forecast_interactive(forecast_df, model, actual_df=None):
    forecast_df = forecast_df.copy()
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    fig = go.Figure()
    # Plot actuals (from forecast file if available, else from actual_df)
    if 'Actual' in forecast_df.columns and not forecast_df['Actual'].isnull().all():
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Actual'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='black', width=2)
        ))
    elif actual_df is not None:
        actual_df = actual_df.copy()
        actual_df['Date'] = pd.to_datetime(actual_df['Date'])
        mask = actual_df['Date'].isin(forecast_df['Date'])
        filtered_actuals = actual_df[mask]
        fig.add_trace(go.Scatter(
            x=filtered_actuals['Date'],
            y=filtered_actuals['Actual'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='black', width=2)
        ))
    # Plot forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Forecast'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='royalblue', dash='dash')
    ))
    fig.update_layout(
        title=f"{model} Forecast vs Actual",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(x=0, y=1),
        hovermode="x unified"
    )
    return fig

def plot_leaderboard_interactive(df):
    fig = px.bar(
        df.sort_values('RMSE'),
        x='RMSE',
        y='Model',
        orientation='h',
        color='RMSE',
        color_continuous_scale='Blues',
        title="Model Leaderboard (Lower RMSE = Better)"
    )
    fig.update_layout(xaxis_title="RMSE", yaxis_title="Model", coloraxis_showscale=False)
    return fig

def plot_monte_carlo_interactive(ticker):
    path = os.path.join(MONTE_CARLO_DIR, ticker, f"{ticker.lower()}_simulations.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    fig = go.Figure()
    for col in df.columns[:100]:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode='lines',
            line=dict(color='skyblue', width=1),
            opacity=0.2,
            showlegend=False
        ))
    fig.update_layout(
        title=f"Monte Carlo Simulation - {ticker} (First 100 Paths)",
        xaxis_title="Days",
        yaxis_title="Simulated Price",
        hovermode="x unified"
    )
    return fig

def plot_mc_distribution_interactive(ticker):
    path = os.path.join(MONTE_CARLO_DIR, ticker, f"{ticker.lower()}_simulations.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    final_prices = df.iloc[-1, :]
    mean = final_prices.mean()
    p5 = final_prices.quantile(0.05)
    p95 = final_prices.quantile(0.95)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=final_prices,
        nbinsx=50,
        name='Final Prices',
        marker_color='skyblue',
        opacity=0.75
    ))
    fig.add_vline(x=mean, line_dash="dash", line_color="blue", annotation_text=f"Mean: {mean:.2f}", annotation_position="top left")
    fig.add_vline(x=p5, line_dash="dash", line_color="red", annotation_text=f"5th %: {p5:.2f}", annotation_position="bottom left")
    fig.add_vline(x=p95, line_dash="dash", line_color="green", annotation_text=f"95th %: {p95:.2f}", annotation_position="top right")
    fig.update_layout(
        title=f"Monte Carlo Final Price Distribution - {ticker}",
        xaxis_title="Simulated Final Price",
        yaxis_title="Count",
        hovermode="x"
    )
    return fig

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def get_rmse_trend(ticker_summary):
    rmse_trend = None
    if 'Timestamp' in ticker_summary.columns and ticker_summary['Timestamp'].nunique() > 1:
        sorted_df = ticker_summary.sort_values('Timestamp')
        if len(sorted_df) >= 2:
            prev = sorted_df.iloc[-2]
            curr = sorted_df.iloc[-1]
            if prev['RMSE'] != 0:
                rmse_trend = (curr['RMSE'] - prev['RMSE']) / abs(prev['RMSE'])
    return rmse_trend

def fmt(val):
    try:
        val = float(val)
        if pd.isna(val):
            return "N/A"
        return f"${val:.2f}"
    except Exception:
        return "N/A"

def executive_summary(ticker, best_model, best_rmse, latest_forecast, latest_actual, mc_mean, mc_p5, mc_p95):
    return (
        f"**{ticker} Executive Summary:**\n\n"
        f"- The best performing model is **{best_model}** with an RMSE of **{fmt(best_rmse)}**.\n"
        f"- The latest forecasted price is **{fmt(latest_forecast)}**.\n"
        f"- The latest actual closing price is **{fmt(latest_actual)}**.\n"
        f"- Monte Carlo simulation suggests an average future price of **{fmt(mc_mean)}**.\n"
        f"- There is a 90% probability the price will end between **{fmt(mc_p5)}** and **{fmt(mc_p95)}**.\n"
    )

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("📊 Executive Stock Forecast Dashboard")

summary_df = load_summary_data()
mc_summary_df = load_mc_summary_data()

if summary_df.empty:
    st.warning("No forecast summary available.")
else:
    tickers = summary_df['Ticker'].unique().tolist()
    selected_ticker = st.selectbox("Select a Ticker:", tickers)

    ticker_summary = summary_df[summary_df['Ticker'] == selected_ticker]
    ticker_summary_sorted = ticker_summary.sort_values('RMSE')
    best_row = ticker_summary_sorted.iloc[0]

    # Load latest forecast for executive KPIs
    forecast_df = load_forecast(selected_ticker, best_row['Model'])
    latest_forecast = float('nan')
    latest_actual = float('nan')
    if forecast_df is not None and not forecast_df.empty:
        forecast_df['Forecast'] = pd.to_numeric(forecast_df['Forecast'], errors='coerce')
        if 'Actual' in forecast_df.columns:
            forecast_df['Actual'] = pd.to_numeric(forecast_df['Actual'], errors='coerce')
            # Get the latest non-NaN value for actual
            actual_non_nan = forecast_df['Actual'].dropna()
            if not actual_non_nan.empty:
                latest_actual = actual_non_nan.iloc[-1]
        # Get the latest non-NaN value for forecast
        forecast_non_nan = forecast_df['Forecast'].dropna()
        if not forecast_non_nan.empty:
            latest_forecast = forecast_non_nan.iloc[-1]

    # Monte Carlo stats for executive KPIs
    mc_stats = None
    mc_mean = mc_p5 = mc_p95 = float('nan')
    if not mc_summary_df.empty and selected_ticker in mc_summary_df['Ticker'].values:
        mc_stats = mc_summary_df[mc_summary_df['Ticker'] == selected_ticker]
        if not mc_stats.empty:
            mc_mean = pd.to_numeric(mc_stats['Mean'].values[0], errors='coerce') if 'Mean' in mc_stats.columns else float('nan')
            mc_p5 = pd.to_numeric(mc_stats['5th Percentile'].values[0], errors='coerce') if '5th Percentile' in mc_stats.columns else float('nan')
            mc_p95 = pd.to_numeric(mc_stats['95th Percentile'].values[0], errors='coerce') if '95th Percentile' in mc_stats.columns else float('nan')

    # === Executive KPI Section ===
    st.markdown("## 🚀 Executive KPIs")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    # Compute latest actual price
    actual_df = load_actuals(selected_ticker)
    latest_actual = float('nan')
    if actual_df is not None and not actual_df.empty:
        actual_df['Date'] = pd.to_datetime(actual_df['Date'])
        value = actual_df.sort_values('Date')['Actual'].iloc[-1]
        try:
            latest_actual = float(value)
        except Exception:
            latest_actual = float('nan')

    kpi1.metric("Best Model", best_row['Model'])
    kpi2.metric("Lowest RMSE", fmt(best_row['RMSE']))
    kpi3.metric("Latest Forecast", fmt(latest_forecast))
    kpi4.metric("Latest Actual", fmt(latest_actual))
    kpi5.metric("MC Mean Price", fmt(mc_mean))

    kpi6, kpi7 = st.columns(2)
    kpi6.metric("MC 5th Percentile", fmt(mc_p5))
    kpi7.metric("MC 95th Percentile", fmt(mc_p95))

    # === RMSE Trend Indicator ===
    rmse_trend = get_rmse_trend(ticker_summary)
    if rmse_trend is not None:
        if rmse_trend < 0:
            st.success(f"RMSE improved by {abs(rmse_trend)*100:.2f}% since last run.")
        elif rmse_trend > 0:
            st.warning(f"RMSE worsened by {abs(rmse_trend)*100:.2f}% since last run.")

    # === Executive Summary Text ===
    st.markdown("---")
    st.markdown(executive_summary(
        selected_ticker, best_row['Model'], best_row['RMSE'], latest_forecast, latest_actual, mc_mean, mc_p5, mc_p95
    ))

    # === Forecast Summary Table ===
    st.markdown("### 📊 Forecast Summary")
    st.dataframe(ticker_summary[['Model', 'MAE', 'MSE', 'RMSE', 'Timestamp']].sort_values('RMSE'))
    csv = convert_df(ticker_summary)
    st.download_button("⬇️ Download Forecast Summary CSV", csv, f"{selected_ticker}_forecast_summary.csv", "text/csv")

    # === Model Leaderboard ===
    st.markdown("---")
    st.markdown("### 🏆 Model Leaderboard")
    leaderboard_fig = plot_leaderboard_interactive(ticker_summary)
    st.plotly_chart(leaderboard_fig, use_container_width=True)

    # === Forecast Charts ===
    st.markdown("---")
    st.markdown("### 📉 Forecast Charts")
    actual_df = load_actuals(selected_ticker)
    latest_actual = float('nan')
    if actual_df is not None and not actual_df.empty:
        actual_df['Date'] = pd.to_datetime(actual_df['Date'])
        # Get the most recent actual price
        latest_actual = actual_df.sort_values('Date')['Actual'].iloc[-1]
    forecast_cols = st.columns(len(ALL_MODELS))
    for i, model in enumerate(ALL_MODELS):
        with forecast_cols[i]:
            forecast_df = load_forecast(selected_ticker, model)
            if forecast_df is not None:
                fig = plot_forecast_interactive(forecast_df, model, actual_df)
                st.plotly_chart(fig, use_container_width=True)

    # === Monte Carlo Simulation ===
    st.markdown("---")
    st.markdown("### 🎲 Monte Carlo Simulation")
    mc_fig = plot_monte_carlo_interactive(selected_ticker)
    if mc_fig:
        st.plotly_chart(mc_fig, use_container_width=True)
    else:
        st.warning("Monte Carlo data not available.")

    # === Monte Carlo Summary Statistics & Distribution ===
    if mc_stats is not None and not mc_stats.empty:
        st.markdown("#### Monte Carlo Summary Statistics")
        st.dataframe(mc_stats)
        csv_mc = convert_df(mc_stats)
        st.download_button("⬇️ Download Monte Carlo Summary CSV", csv_mc, f"{selected_ticker}_monte_carlo_summary.csv", "text/csv")

        st.markdown("#### Distribution of Final Simulated Prices")
        mc_dist_fig = plot_mc_distribution_interactive(selected_ticker)
        st.plotly_chart(mc_dist_fig, use_container_width=True)
