import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import os

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_forecasts(y_true, arima_pred, lstm_pred, ticker="AAPL"):
    results = {}

    results["ARIMA"] = {
        "MAE": mean_absolute_error(y_true, arima_pred),
        "RMSE": root_mean_squared_error(y_true, arima_pred),
        "MAPE": mean_absolute_percentage_error(y_true, arima_pred)
    }

    results["LSTM"] = {
        "MAE": mean_absolute_error(y_true, lstm_pred),
        "RMSE": root_mean_squared_error(y_true, lstm_pred),
        "MAPE": mean_absolute_percentage_error(y_true, lstm_pred)
    }

    print(f"\n\U0001F4CA Forecast Evaluation for {ticker}:")
    for model, metrics in results.items():
        print(f"\n\U0001F539 {model} Model:")
        for name, val in metrics.items():
            print(f"   {name}: {val:.4f}")

    return results

def plot_forecasts(y_true, arima_pred, lstm_pred, ticker="AAPL"):
    days = np.arange(len(y_true))
    plt.figure(figsize=(12, 6))
    plt.plot(days, y_true, label="Actual", linewidth=2)
    plt.plot(days, arima_pred, label="ARIMA Forecast", linestyle="--")
    plt.plot(days, lstm_pred, label="LSTM Forecast", linestyle=":")
    plt.title(f"{ticker} - Forecast vs Actual")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_error_metrics(results, ticker="AAPL"):
    labels = list(results.keys())
    metrics = list(results[labels[0]].keys())

    data = {metric: [results[model][metric] for model in labels] for metric in metrics}
    x = np.arange(len(labels))
    width = 0.25

    _, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, data[metric], width, label=metric)

    ax.set_ylabel('Score')
    ax.set_title(f'{ticker} Forecast Error Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    forecast_days = 30
    ticker = "AAPL"

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    true_path = os.path.join(data_dir, "true_prices.csv")
    arima_path = os.path.join(data_dir, "arima_forecast.csv")
    lstm_path = os.path.join(data_dir, "lstm_forecast.csv")

    # Check if files exist before proceeding
    for path in [true_path, arima_path, lstm_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    y_true = pd.read_csv(true_path).values[-forecast_days:].flatten()
    arima_pred = pd.read_csv(arima_path).values[-forecast_days:].flatten()
    lstm_pred = pd.read_csv(lstm_path).values[-forecast_days:].flatten()

    results = evaluate_forecasts(y_true, arima_pred, lstm_pred, ticker)
    plot_forecasts(y_true, arima_pred, lstm_pred, ticker)
    plot_error_metrics(results, ticker)
