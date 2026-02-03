"""
Evaluation metrics for forecasting.
"""

"""
Evaluation metrics for multivariate time series forecasting models.
"""

import torch

def mae(preds, targets):
    return torch.mean(torch.abs(preds - targets)).item()

def rmse(preds, targets):
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()
