import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_mase(y_true, y_pred, y_train, seasonality=1):
    # MASE requires the training data to calculate the naive error scale
    naive_error = np.mean(np.abs(np.diff(y_train, n=seasonality)))
    mae = mean_absolute_error(y_true, y_pred)
    return mae / naive_error

def calculate_crps(y_true, quantiles):
    # Simplified CRPS for 10th, 50th, 90th quantiles
    # quantiles: [N, 3] -> (q10, q50, q90)
    q10, q50, q90 = quantiles[:, 0], quantiles[:, 1], quantiles[:, 2]
    e50 = np.abs(y_true - q50)
    # Average pinning loss for the quantiles
    def pinball_loss(y, q, tau):
        err = y - q
        return np.maximum(tau * err, (tau - 1) * err).mean()
    
    return (pinball_loss(y_true, q10, 0.1) + pinball_loss(y_true, q50, 0.5) + pinball_loss(y_true, q90, 0.9)) / 3

def calculate_metrics(y_true, y_pred, y_train=None, quantiles=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    res = {"MAE": mae, "RMSE": rmse}
    
    if y_train is not None:
        res["MASE"] = calculate_mase(y_true, y_pred, y_train)
    if quantiles is not None:
        res["CRPS"] = calculate_crps(y_true, quantiles)
        
    return res