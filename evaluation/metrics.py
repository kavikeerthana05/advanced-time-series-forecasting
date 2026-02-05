"""
Evaluation metrics for forecasting.
"""

"""
Evaluation metrics for multivariate time series forecasting models.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # R2 Score is a good proxy for "accuracy" in regression
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}