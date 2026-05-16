import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values using forward fill then backward fill."""
    return df.ffill().bfill()

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds RSI, MACD, and Bollinger Bands to the dataframe."""
    df = df.copy()
    if 'Close' not in df.columns:
        return df

    # RSI (14 days)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_Band'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()

    # Lag features
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_5'] = df['Close'].shift(5)

    return df.dropna()

def get_log_returns(df: pd.DataFrame) -> pd.Series:
    """Calculates log returns of the 'Close' column."""
    return np.log(df['Close'] / df['Close'].shift(1)).dropna()
