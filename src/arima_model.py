import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

def load_stock_data(ticker="AAPL", data_path="../data/"):
    """
    Load stock data for a given ticker from CSV files.
    
    Args:
        ticker (str): Stock ticker symbol.
        data_path (str): Path to the directory containing CSV files.
        
    Returns:
        pd.Series: Series with Date as index and closing prices.
    """
    file_path = os.path.join(data_path, f"{ticker}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file for {ticker} not found at {file_path}")
    
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Try to find the most appropriate column for closing price
    possible_close_cols = ["Close", "close", ticker.upper()]
    close_col = next((col for col in possible_close_cols if col in df.columns), None)

    if not close_col:
        raise ValueError(f"No valid price column (e.g., 'Close', '{ticker}') found in {file_path}")

    series = pd.to_numeric(df[close_col], errors="coerce").dropna()
    return series

def test_stationarity(series):
    """
    Perform Augmented Dickey-Fuller test and print results.
    """
    series = series.dropna()

    if series.empty:
        raise ValueError("The time series is empty after cleaning. Check input data.")

    result = adfuller(series)
    print(f"\nADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    return result[1] < 0.05  # Stationary if p < 0.05

def fit_arima(series, order=(1, 1, 1)):
    """
    Fit an ARIMA model to the time series.
    """
    model = ARIMA(series, order=order)
    fitted = model.fit()
    print("\nModel Summary:")
    print(fitted.summary())
    return fitted

def forecast_and_plot(series, model_fit, forecast_days=30, ticker="AAPL"):
    """
    Forecast future values and plot the results.
    """
    forecast = model_fit.forecast(steps=forecast_days)
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, label="Historical", color="blue")

    future_dates = pd.date_range(start=series.index[-1], periods=forecast_days+1, freq='B')[1:]
    plt.plot(future_dates, forecast, label="Forecast", color="red")

    plt.title(f"{ticker} Stock Price Forecast ({forecast_days} Days)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main(ticker="AAPL", forecast_days=30, arima_order=(1, 1, 1), data_path="../data/"):
    """
    Run the ARIMA forecasting pipeline for a given stock ticker.
    """
    try:
        print(f"Loading data for {ticker}...")
        series = load_stock_data(ticker, data_path)

        print("Testing stationarity...")
        stationary = test_stationarity(series)

        if stationary:
            print("Series is stationary. Proceeding with ARIMA.")
        else:
            print("Series is non-stationary. Proceeding with differencing (d=1).")

        print("Fitting ARIMA model...")
        model_fit = fit_arima(series, order=arima_order)

        print(f"Forecasting next {forecast_days} days...")
        forecast_and_plot(series, model_fit, forecast_days, ticker)

    except Exception as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    # You can change these values as needed
    main(ticker="MSFT", forecast_days=30, arima_order=(1, 1, 1), data_path="../data/")
