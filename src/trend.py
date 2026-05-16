import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

def perform_stl_decomposition(series: pd.Series, period=252):
    """
    Performs Seasonal-Trend decomposition.
    period=252 corresponds to roughly one trading year.
    """
    # Fill NaN values to ensure decomposition works
    series = series.ffill().bfill()
    result = seasonal_decompose(series, model='multiplicative', period=period)
    
    df = pd.DataFrame({
        'Observed': result.observed,
        'Trend': result.trend,
        'Seasonal': result.seasonal,
        'Residual': result.resid
    })
    return df
