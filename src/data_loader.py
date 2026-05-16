import yfinance as yf
import pandas as pd
import os
import logging
from src.utils import setup_logger

logger = logging.getLogger(__name__)

class StockDataLoader:
    def __init__(self, tickers: list, start_date: str, end_date: str, raw_dir: str = "data/raw"):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.raw_dir = raw_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        
    def fetch_and_save(self):
        """Fetches data from Yahoo Finance and saves as CSV."""
        logger.info(f"Fetching data for {len(self.tickers)} tickers from {self.start_date} to {self.end_date}")
        all_data = {}
        for ticker in self.tickers:
            try:
                logger.info(f"Downloading {ticker}...")
                data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                if data.empty:
                    logger.warning(f"No data found for {ticker}")
                    continue
                
                # yfinance 1.3.0+ returns a MultiIndex column structure: (Price, Ticker)
                if isinstance(data.columns, pd.MultiIndex):
                    # We want to keep only the 'Price' level (e.g., 'Open', 'High', 'Close')
                    data.columns = data.columns.get_level_values(0)
                
                filepath = os.path.join(self.raw_dir, f"{ticker}.csv")
                data.to_csv(filepath)
                all_data[ticker] = data
                logger.info(f"Saved {ticker} to {filepath}")
            except Exception as e:
                logger.error(f"Failed to download {ticker}: {e}")
        return all_data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    TICKERS = [
        "TCS.NS", 
        "HDFCBANK.NS", 
        "RELIANCE.NS", 
        "SUNPHARMA.NS", 
        "HINDUNILVR.NS", 
        "TATAMOTORS.NS"
    ]
    loader = StockDataLoader(TICKERS, "2021-01-01", "2025-12-31")
    loader.fetch_and_save()
