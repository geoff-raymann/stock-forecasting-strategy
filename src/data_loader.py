# src/data_loader.py

import yfinance as yf
import os
import pandas as pd
import subprocess
import logging
import smtplib
from email.message import EmailMessage

# === CONFIG ===
TICKERS = ['AAPL', 'MSFT', 'PFE', 'JNJ']  # Add more if needed
START_DATE = '2019-01-01'
END_DATE = None  # Default = today
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
LOG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_loader.log'))

# === EMAIL SETTINGS (optional) ===
ENABLE_EMAIL = True  # Set to True to enable email notifications
EMAIL_FROM = 'odiwuorgeoff50@gmail.com'
EMAIL_TO = 'odiwuorgeoff50@gmail.com'
EMAIL_PASS = 'dnfl pnrj btzc cazu'  # Use an app password for Gmail

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def send_notification(subject, body):
    if not ENABLE_EMAIL:
        return
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_FROM, EMAIL_PASS)
            smtp.send_message(msg)
        logging.info("Notification email sent.")
    except Exception as e:
        logging.error(f"Failed to send notification email: {e}")

def fetch_ticker_data(ticker: str, start=START_DATE, end=END_DATE):
    logging.info(f"Fetching {ticker} data from Yahoo Finance...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)

    if df.empty:
        logging.warning(f"No data returned for {ticker}")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(out_path)
    logging.info(f"Saved {ticker} to {out_path}")

def main():
    for ticker in TICKERS:
        fetch_ticker_data(ticker)

if __name__ == '__main__':
    try:
        main()
        subprocess.run(["python", "src/batch_runner.py"], check=True)
        send_notification(
            "Stock Data Loader Success",
            "Data loading and batch processing completed successfully."
        )
    except Exception as e:
        logging.exception("Script failed!")
        send_notification(
            "Stock Data Loader Failure",
            f"Script failed with error: {e}"
        )
