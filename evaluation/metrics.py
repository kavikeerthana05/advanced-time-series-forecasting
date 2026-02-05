"""
Evaluation metrics for forecasting.
"""

"""
Evaluation metrics for multivariate time series forecasting models.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculates comprehensive metrics for time series forecasting.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Mean Model Error (MME) - often used in specific assignments
    mme = np.mean(np.abs((y_true - y_pred) / (np.mean(y_true) + 1e-5)))
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MME": mme}