import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.portfolio import optimize_portfolio
from src.preprocessing import get_log_returns
from src.volatility import calculate_rolling_volatility
from src.trend import perform_stl_decomposition
from src.forecasting import run_arima_pipeline

# Optional TensorFlow import
try:
    from src.forecasting import run_lstm_pipeline
    LSTM_AVAILABLE = True
except:
    LSTM_AVAILABLE = False

def load_data():
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
        return {}
    data = {}
    for f in os.listdir(raw_dir):
        if f.endswith('.csv'):
            ticker = f.split('.')[0]
            df = pd.read_csv(os.path.join(raw_dir, f), index_col='Date', parse_dates=True)
            data[ticker] = df
    return data

def main():
    st.set_page_config(page_title="StockGro Capstone Engine", layout="wide", page_icon="📈")
    
    st.markdown("""
        <style>
        .reportview-container { background: #0E1117; }
        .sidebar .sidebar-content { background: #262730; }
        h1 { color: #00FFAA; font-family: 'Inter', sans-serif; text-align: center; }
        .metric-card {
            background-color: #1E1E1E; padding: 20px; border-radius: 10px;
            border-left: 5px solid #00FFAA; margin-bottom: 20px; text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Quantitative AI Portfolio Platform")
    st.markdown("*End-to-End Analytics, Forecasting, and StockGro Execution Validation*")

    data = load_data()
    if not data:
        st.warning("No data found in data/raw. Please run the data loader pipeline.")
        return

    st.sidebar.header("Controls")
    selected_tickers = st.sidebar.multiselect("Select Portfolio Assets", list(data.keys()), default=list(data.keys()))

    if not selected_tickers:
        st.info("Please select at least one asset.")
        return

    # Create Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Portfolio Analytics", 
        "📉 Volatility & Trend", 
        "🤖 AI Forecasting", 
        "⚖️ Optimization",
        "📸 StockGro Execution"
    ])

    # ---------------------------
    # TAB 1: Portfolio Analytics
    # ---------------------------
    with tab1:
        st.header("Asset Performance")
        prices = pd.DataFrame({t: data[t]['Close'] for t in selected_tickers}).dropna()
        normalized = (prices / prices.iloc[0]) * 100
        fig_norm = px.line(normalized, title="Normalized Growth (Base 100)", template="plotly_dark")
        st.plotly_chart(fig_norm, use_container_width=True)

        st.header("Inter-Asset Correlation")
        log_returns = pd.DataFrame({t: get_log_returns(data[t]) for t in selected_tickers}).dropna()
        corr_matrix = log_returns.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation Heatmap", template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)

    # ---------------------------
    # TAB 2: Volatility & Trend
    # ---------------------------
    with tab2:
        st.header("Macro Trend Extraction (STL Decomposition)")
        trend_ticker = st.selectbox("Select Asset for STL", selected_tickers, key="stl_tick")
        stl_df = perform_stl_decomposition(data[trend_ticker]['Close'])
        
        fig_stl = go.Figure()
        fig_stl.add_trace(go.Scatter(x=stl_df.index, y=stl_df['Observed'], name="Observed", line=dict(color='gray', width=1)))
        fig_stl.add_trace(go.Scatter(x=stl_df.index, y=stl_df['Trend'], name="Trend", line=dict(color='#00FFAA', width=3)))
        fig_stl.update_layout(title=f"Underlying Macro Trend for {trend_ticker}", template="plotly_dark")
        st.plotly_chart(fig_stl, use_container_width=True)

        st.header("Annualized Rolling Volatility")
        rolling_vol = pd.DataFrame({t: calculate_rolling_volatility(log_returns[t]) for t in selected_tickers}).dropna()
        fig_vol = px.line(rolling_vol, title="30-Day Rolling Volatility Risk", template="plotly_dark")
        st.plotly_chart(fig_vol, use_container_width=True)

    # ---------------------------
    # TAB 3: AI Forecasting
    # ---------------------------
    with tab3:
        st.header("Time-Series Forecast Validation")
        col_f1, col_f2 = st.columns(2)
        target_ticker = col_f1.selectbox("Target Asset", selected_tickers, key="fcast_tick")
        models = ["ARIMA"]

        if LSTM_AVAILABLE:
            models.append("LSTM Neural Network")

        model_type = col_f2.radio("Model Architecture", models)
        
        if st.button("Train & Forecast Pipeline"):
            series = data[target_ticker]['Close'].dropna()
            with st.spinner(f"Training {model_type} on {target_ticker}... This may take a moment."):
                if model_type == "ARIMA":
                    df_pred, metrics = run_arima_pipeline(series, test_size=60)
                elif LSTM_AVAILABLE:
                    df_pred, metrics = run_lstm_pipeline(series, epochs=3)
                else:
                    st.error("LSTM model unavailable in deployment environment.")
                    st.stop()# Low epochs for fast web demo
                
                # Plot
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=df_pred['Date'], y=df_pred['Actual'], name="Actual Price", line=dict(color='#00FFAA')))
                fig_pred.add_trace(go.Scatter(x=df_pred['Date'], y=df_pred['Predicted'], name="Predicted Price", line=dict(color='#FF0055', dash='dash')))
                if 'Lower_CI' in df_pred.columns:
                    fig_pred.add_trace(go.Scatter(x=df_pred['Date'], y=df_pred['Upper_CI'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                    fig_pred.add_trace(go.Scatter(x=df_pred['Date'], y=df_pred['Lower_CI'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', fillcolor='rgba(255, 0, 85, 0.2)', name='95% CI'))
                
                fig_pred.update_layout(title=f"{model_type} 60-Day Forecast Evaluation for {target_ticker}", template="plotly_dark")
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Metrics Scorecard
                st.markdown("### Evaluation Metrics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("RMSE", f"₹{metrics['RMSE']:.2f}")
                m2.metric("MAE", f"₹{metrics['MAE']:.2f}")
                m3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                m4.metric("Directional Acc.", f"{metrics['DA']:.2f}%")

    # ---------------------------
    # TAB 4: Portfolio Optimizer
    # ---------------------------
    with tab4:
        st.header("Modern Portfolio Theory (Mean-Variance)")
        mean_ret = log_returns.mean()
        cov_mat = log_returns.cov()

        if st.button("Generate Optimal Weights"):
            with st.spinner("Maximizing Sharpe Ratio..."):
                weights = optimize_portfolio(mean_ret.values, cov_mat.values)
                fig_pie = px.pie(values=weights, names=selected_tickers, title="Max Sharpe Allocation", template="plotly_dark", hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
                
                ret = np.sum(mean_ret.values * weights) * 252
                vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat.values, weights))) * np.sqrt(252)
                
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"<div class='metric-card'><h4>Exp. Return</h4><h2>{ret*100:.2f}%</h2></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-card'><h4>Volatility</h4><h2>{vol*100:.2f}%</h2></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric-card'><h4>Sharpe</h4><h2>{ret/vol:.2f}</h2></div>", unsafe_allow_html=True)

    # ---------------------------
    # TAB 5: StockGro Execution
    # ---------------------------
    with tab5:
        st.header("Live Simulated Trading Evidence")
        st.markdown("Authentic screenshots from the ₹10,00,000 StockGro execution phase, validating the portfolio strategy.")
        
        ss_dir = "screenshots"
        if os.path.exists(ss_dir):
            images = [f for f in os.listdir(ss_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                # Display in a grid
                cols = st.columns(3)
                for idx, img_name in enumerate(images):
                    img_path = os.path.join(ss_dir, img_name)
                    cols[idx % 3].image(img_path, caption=img_name, use_column_width=True)
            else:
                st.info("No images found in the screenshots directory.")

if __name__ == "__main__":
    main()
