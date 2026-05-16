import pandas as pd
import numpy as np
from arch import arch_model

def calculate_rolling_volatility(returns: pd.Series, window=30) -> pd.Series:
    """Calculates annualized rolling volatility."""
    # Assuming 252 trading days
    return returns.rolling(window=window).std() * np.sqrt(252)

def fit_garch(returns: pd.Series):
    """Fits a GARCH(1,1) model to the returns."""
    # Rescale returns for better optimization convergence
    rescaled_returns = returns * 100
    model = arch_model(rescaled_returns, vol='Garch', p=1, q=1, rescale=False)
    results = model.fit(disp='off')
    return results
