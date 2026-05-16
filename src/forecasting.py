import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
try:
    import tensorflow as tf
except:
    tf = None
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    TENSORFLOW_AVAILABLE = True
except:
    TENSORFLOW_AVAILABLE = False
from src.evaluation import evaluate_forecast

def run_arima_pipeline(series: pd.Series, test_size=30, order=(5,1,0)):
    """Runs full ARIMA pipeline: split, train, forecast, evaluate."""
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    
    model = ARIMA(train, order=order)
    fitted_model = model.fit()
    
    forecast_obj = fitted_model.get_forecast(steps=test_size)
    forecast_mean = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()
    
    metrics = evaluate_forecast(test.values, forecast_mean.values)
    
    df = pd.DataFrame({
        'Date': test.index,
        'Actual': test.values,
        'Predicted': forecast_mean.values,
        'Lower_CI': conf_int.iloc[:, 0].values,
        'Upper_CI': conf_int.iloc[:, 1].values
    })
    
    return df, metrics

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def run_lstm_pipeline(series: pd.Series, seq_length=60, epochs=5, batch_size=32):
    """Runs full LSTM pipeline: scale, split, train, forecast, evaluate."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    
    X, y = create_sequences(scaled_data, seq_length)
    
    # 80/20 Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Use quiet verbosity for dashboard execution
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), verbose=0)
    
    predictions = model.predict(X_test, verbose=0)
    
    # Inverse transform
    predictions_inv = scaler.inverse_transform(predictions).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    metrics = evaluate_forecast(y_test_inv, predictions_inv)
    
    # Map back to dates
    test_dates = series.index[-len(y_test_inv):]
    
    df = pd.DataFrame({
        'Date': test_dates,
        'Actual': y_test_inv,
        'Predicted': predictions_inv
    })
    
    return df, metrics
