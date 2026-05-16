import numpy as np
import pandas as pd
from scipy.optimize import minimize

def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculates portfolio return and volatility."""
    returns = np.sum(mean_returns * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    """Objective function for maximizing Sharpe ratio."""
    p_ret, p_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_vol

def optimize_portfolio(mean_returns, cov_matrix):
    """Finds optimal portfolio weights using Max Sharpe Ratio."""
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    
    # Initial guess (equal weighting)
    init_guess = num_assets * [1. / num_assets,]
    
    result = minimize(negative_sharpe_ratio, init_guess, args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x
