# Capstone Project: Data-Driven Stock Analysis & Portfolio Optimization on StockGro

## 📊 Overview
This is a production-ready Quantitative Finance platform. It fetches authentic historical stock data from the National Stock Exchange (NSE), conducts complex time-series forecasting (ARIMA & LSTM), dynamically calculates mathematical risk (GARCH & Rolling Volatility), and computes the mathematically optimal portfolio allocation (Max Sharpe Ratio). 

Crucially, this project transcends academic theory by proving execution via **real-world simulated trading on the StockGro platform**, validated by comprehensive screenshot evidence.

## 🏗️ Architecture

```text
project/
├── data/raw/              # Live NSE CSV datasets fetched via yfinance
├── notebooks/             # Iterative ML prototyping
├── src/                   # The Core Engine
│   ├── data_loader.py     # YFinance MultiIndex ingestion
│   ├── preprocessing.py   # Log returns, RSI, MACD
│   ├── forecasting.py     # ARIMA/LSTM pipelines with train/test splits
│   ├── volatility.py      # Rolling Volatility
│   ├── trend.py           # STL Seasonal-Trend Decomposition
│   ├── portfolio.py       # Mean-Variance Optimization
│   └── evaluation.py      # RMSE, MAE, MAPE, Directional Accuracy
├── dashboard/app.py       # Multi-Tab Streamlit Interactive UI
├── reports/               # Final Markdown Documentation
└── screenshots/           # Authentic StockGro execution evidence
```

## 🚀 Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/Asterioxer/stockgro.git
cd stockgro
```

**2. Create a Virtual Environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Fetch Real Market Data**
```bash
python -m src.data_loader
```

## 🖥️ Usage (Streamlit Dashboard)
To run the live interactive quant dashboard, execute:
```bash
streamlit run dashboard/app.py
```
This will open a 5-tab web application where you can explore Asset Analytics, Volatility, AI Forecasts, Portfolio Optimization, and view the raw StockGro Execution screenshots.

## 📈 Methodology
- **Time Series Forecasting**: Utilizes ARIMA for linear trend projection and Deep LSTM Networks for discovering non-linear sequence dependencies. Evaluated rigorously using RMSE and Directional Accuracy.
- **Modern Portfolio Theory**: Computes the Covariance Matrix of asset log returns to generate the "Efficient Frontier", securing the highest possible return per unit of risk.
- **Real-World Testing**: The mathematically determined weights were executed in a virtual ₹10,00,000 StockGro environment.
