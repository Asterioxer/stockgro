# Executive Summary: Data-Driven Stock Analysis

## Objective
To build a production-ready quantitative trading engine that forecasts stock prices using ARIMA and LSTM, analyzes volatility via GARCH, and dynamically optimizes portfolio allocations to maximize the Sharpe Ratio.

## Methodology
1. **Data Collection**: 10 sector-diversified NSE stocks (2021-2025).
2. **Feature Engineering**: RSI, MACD, Bollinger Bands, and Log Returns.
3. **Forecasting**:
   - ARIMA for linear time series trends.
   - LSTM for complex non-linear sequence modeling.
4. **Portfolio Optimization**: Mean-Variance Optimization for risk-adjusted returns.
5. **Dashboard**: Interactive Streamlit interface.

## System Architecture
Our codebase is modularized (`src/`) to separate exploratory research (`notebooks/`) from production logic. This structure is essential for deploying algorithmic trading systems.

## Conclusion
The system successfully integrates predictive modeling with Modern Portfolio Theory, allowing a simulated ₹10,00,000 portfolio on StockGro to be allocated strictly based on data-driven forecasts.
