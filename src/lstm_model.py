import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def load_data(file_path, column=None):
    df = pd.read_csv(file_path)
    
    if column is None or column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        print("⚠️ Provided column is missing or non-numeric. Attempting to auto-detect...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            raise ValueError("❌ No numeric columns found in the CSV.")
        column = numeric_cols[0]
        print(f"✅ Auto-selected column: {column}")
    
    series = df[column].dropna().astype(float).values.reshape(-1, 1)
    return series, column

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast(model, last_seq, forecast_days):
    predictions = []
    seq = last_seq.copy()
    for _ in range(forecast_days):
        pred = model.predict(seq.reshape(1, seq.shape[0], 1), verbose=0)[0][0]
        predictions.append(pred)
        seq = np.append(seq[1:], [[pred]], axis=0)
    return np.array(predictions)

def main(args):
    # Load and preprocess data
    series, column_used = load_data(args.file, args.column)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series)
    X, y = create_sequences(scaled, args.seq_len)

    # Build and train model
    model = build_lstm((X.shape[1], 1))
    model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    # Forecast
    last_seq = scaled[-args.seq_len:]
    preds = forecast(model, last_seq, args.forecast)
    forecasted = scaler.inverse_transform(preds.reshape(-1, 1))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(series[-100:], label="Historical (Last 100 Days)")
    plt.plot(np.arange(100, 100 + args.forecast), forecasted, label="LSTM Forecast", color='red')
    plt.title(f"{column_used} Price Forecast (LSTM)")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Financial Time Series Forecasting")
    parser.add_argument('--file', type=str, required=True, help='CSV file path')
    parser.add_argument('--column', type=str, default=None, help='Column name for price series')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length (history window)')
    parser.add_argument('--forecast', type=int, default=30, help='Forecast horizon (days)')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    main(args)
