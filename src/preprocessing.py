import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_stock_data(
    data_path="../data/",
    tickers=None,
    price_column="Close",
    dropna=True,
    verbose=True
):
    """
    Loads and merges stock price data for given tickers from CSV files.

    Args:
        data_path (str): Path to the directory containing CSV files.
        tickers (list): List of ticker symbols.
        price_column (str): Column name to extract from each CSV.
        dropna (bool): Whether to drop rows with missing values.
        verbose (bool): Print warnings if files are missing.

    Returns:
        pd.DataFrame: Merged DataFrame with price_column for each ticker.
    """
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'JNJ', 'PFE']

    data = {}
    for ticker in tickers:
        file_path = os.path.join(data_path, f"{ticker}.csv")
        if not os.path.exists(file_path):
            if verbose:
                print(f"Warning: File not found for {ticker}: {file_path}")
            continue
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        if price_column not in df.columns:
            if verbose:
                print(f"Warning: Column '{price_column}' not found in {file_path}")
            continue
        data[ticker] = df[[price_column]].rename(columns={price_column: ticker})

    if data:
        df_all = pd.concat(data.values(), axis=1)
        if dropna:
            df_all.dropna(inplace=True)
        return df_all
    else:
        if verbose:
            print("No data loaded. Please check if the CSV files exist in the data directory.")
        return pd.DataFrame()

def set_plot_style():
    available_styles = plt.style.available
    if "seaborn-darkgrid" in available_styles:
        plt.style.use("seaborn-darkgrid")
    elif "seaborn-v0_8-darkgrid" in available_styles:
        plt.style.use("seaborn-v0_8-darkgrid")
    else:
        plt.style.use("default")
    sns.set_palette("muted")

# Example usage:
if __name__ == "__main__":
    set_plot_style()
    tickers = ['AAPL', 'MSFT', 'JNJ', 'PFE']
    df_all = load_stock_data(tickers=tickers)
    print(df_all.head())
