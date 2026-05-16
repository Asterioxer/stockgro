# Final Capstone Report: Data-Driven Stock Analysis

## 1. Methodology & Stock Selection Rationale
To construct a resilient portfolio, we selected a highly diversified universe of large-cap National Stock Exchange (NSE) assets spanning fundamentally disparate sectors:
- **IT**: `TCS.NS`
- **Banking**: `HDFCBANK.NS`
- **Energy**: `RELIANCE.NS`
- **Pharma**: `SUNPHARMA.NS`
- **FMCG**: `HINDUNILVR.NS`

Diversification is the only "free lunch" in finance. By picking uncorrelated assets, we minimize systematic sector risk. We ingested historical daily data from 2021-01-01 to 2025-12-31.

## 2. Volatility & Trend Analysis
Before deploying AI, we applied **STL Decomposition** to extract the true macroeconomic drift of the assets, stripping away seasonal noise. We computed **Log Returns** (to achieve mathematical stationarity) and **30-Day Annualized Rolling Volatility** to understand the historical risk premium of the assets. 

## 3. Forecasting Models
We deployed two distinct modeling paradigms:
1. **ARIMA (AutoRegressive Integrated Moving Average)**: Acts as our linear baseline. It differs the data to achieve stationarity and projects the linear trend forward with 95% confidence intervals.
2. **LSTM (Long Short-Term Memory)**: A Deep Learning Recurrent Neural Network. We implemented an 80/20 train-test temporal split. The network processes 60-day historical sequences (`seq_length=60`) to capture hidden, non-linear market dependencies. We utilized Dropout layers (20%) to prevent overfitting to market noise.

**Evaluation**: We strictly evaluated both models using RMSE (penalizing large errors), MAE, MAPE, and **Directional Accuracy** (the metric most critical to trading profitability).

## 4. Portfolio Strategy
Using the authentic historical covariance matrix, we ran a **Mean-Variance Optimization** algorithm. This bounded optimization maximizes the **Sharpe Ratio**, effectively returning the exact percentage weighting for each stock that yields the highest expected return per unit of risk.

## 5. StockGro Execution & Reflection
The theoretical weights calculated by our AI engine were manually executed on the **StockGro** platform using a virtual ₹10,00,000 portfolio. 
- The execution is fully documented in the `/screenshots` directory embedded within the Streamlit dashboard.
- **Reflection**: Translating quantitative weights into live trades highlighted the impact of slippage and execution timing. While the LSTM provided directional intuition, the strict Mean-Variance portfolio acted as the true risk-management anchor. Future iterations could explore Reinforcement Learning to automate the execution phase entirely.
