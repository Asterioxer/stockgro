import pandas as pd
import numpy as np
import os

TICKERS = [
    "TCS.NS", "INFY.NS",
    "HDFCBANK.NS", "ICICIBANK.NS",
    "RELIANCE.NS", "ONGC.NS",
    "ITC.NS", "HINDUNILVR.NS",
    "SUNPHARMA.NS", "CIPLA.NS"
]

def generate_synthetic_data(ticker, start="2021-01-01", end="2025-12-31"):
    dates = pd.date_range(start=start, end=end, freq='B')
    n = len(dates)
    
    np.random.seed(hash(ticker) % (2**32))
    
    # Geometric Brownian Motion
    S0 = np.random.uniform(500, 3000)
    mu = 0.1 / 252
    sigma = 0.2 / np.sqrt(252)
    
    returns = np.random.normal(mu, sigma, n)
    price = S0 * np.exp(np.cumsum(returns))
    
    # Add High, Low, Open
    open_p = price * np.random.normal(1, 0.005, n)
    high_p = np.maximum(open_p, price) * np.random.normal(1.005, 0.002, n)
    low_p = np.minimum(open_p, price) * np.random.normal(0.995, 0.002, n)
    volume = np.random.randint(100000, 5000000, n)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_p,
        'High': high_p,
        'Low': low_p,
        'Close': price,
        'Adj Close': price,
        'Volume': volume
    })
    df.set_index('Date', inplace=True)
    return df

def main():
    os.makedirs('data/raw', exist_ok=True)
    for ticker in TICKERS:
        df = generate_synthetic_data(ticker)
        df.to_csv(f'data/raw/{ticker}.csv')
        print(f"Generated synthetic data for {ticker}")

if __name__ == "__main__":
    main()
