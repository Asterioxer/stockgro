import numpy as np

def calculate_rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mae(y_true, y_pred):
    """Calculates Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_directional_accuracy(y_true, y_pred):
    """Calculates Directional Accuracy."""
    actual_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    
    # Handle cases where diff is 0 (no movement)
    actual_direction = np.where(actual_direction == 0, 1, actual_direction)
    pred_direction = np.where(pred_direction == 0, 1, pred_direction)
    
    match = (actual_direction == pred_direction)
    return np.mean(match) * 100

def evaluate_forecast(y_true, y_pred):
    """Returns a dictionary of all evaluation metrics."""
    return {
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": calculate_mae(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
        "DA": calculate_directional_accuracy(y_true, y_pred)
    }
