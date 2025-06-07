import os
import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker: str, start_date: str, end_date: str, save_csv: bool = True, data_dir: str = "data") -> pd.DataFrame:
    """
    Downloads historical stock data from Yahoo Finance using yfinance.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        save_csv (bool): Whether to save the data as a CSV file.
        data/raw_dir (str): Directory to save CSV file if save_csv is True.

    Returns:
        pd.DataFrame: DataFrame containing historical stock data.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")

    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        print(f"No data found for {ticker}.")
        return df

    df.reset_index(inplace=True)
    df.dropna(inplace=True)

    if save_csv:
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, f"{ticker}_stock_data.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved {ticker} data to {file_path}")

    return df


def fetch_multiple_stocks(tickers: list, start_date: str, end_date: str, save_csv: bool = True, data_dir: str = "data") -> dict:
    """
    Fetches historical data for multiple stock tickers.

    Parameters:
        tickers (list): List of stock tickers.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        save_csv (bool): Whether to save each stock as a CSV file.
        data/raw_dir (str): Directory to save CSV files.

    Returns:
        dict: Dictionary of {ticker: DataFrame}
    """
    data = {}
    for ticker in tickers:
        df = fetch_stock_data(ticker, start_date, end_date, save_csv, data_dir)
        if not df.empty:
            data[ticker] = df
    return data


if __name__ == "__main__":
    # Example use case
    tickers = ["AAPL", "MSFT", "JNJ", "PFE"]  # Tech and healthcare
    start = "2015-01-01"
    end = "2024-12-31"
    fetch_multiple_stocks(tickers, start, end)
