import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for synthetic time-series forecasting.

    Each sample consists of:
    - Input sequence of length `input_window`
    - Target sequence of length `forecast_horizon`

    The generated time series includes:
    - Linear trend
    - Seasonal (sinusoidal) component
    - Long-term dependency (lagged influence)
    - Gaussian noise
    """

    def __init__(self, series, input_window, forecast_horizon):
        """
        Args:
            series (np.ndarray): Time series data of shape (T, 1)
            input_window (int): Number of past timesteps used as input
            forecast_horizon (int): Number of future timesteps to predict
        """
        self.series = torch.tensor(series.values, dtype=torch.float32)
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.series) - self.input_window - self.forecast_horizon

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.input_window]
        y = self.series[
            idx + self.input_window :
            idx + self.input_window + self.forecast_horizon
        ]
        return x, y


def generate_synthetic_series(
    length=2000,
    trend_strength=0.01,
    seasonality_strength=1.0,
    seasonality_period=50,
    noise_std=0.1,
    long_term_lag=25,
    long_term_weight=0.5,
):
    """
    Generates a synthetic time series with trend, seasonality,
    long-term dependency, and noise.

    Args:
        length (int): Total length of time series
        trend_strength (float): Linear trend slope
        seasonality_strength (float): Amplitude of seasonality
        seasonality_period (int): Period of sinusoidal component
        noise_std (float): Standard deviation of Gaussian noise
        long_term_lag (int): Lag for long-term dependency
        long_term_weight (float): Weight of lagged contribution

    Returns:
        np.ndarray: Generated time series of shape (length, 1)
    """

    t = np.arange(length)

    # Trend
    trend = trend_strength * t

    # Seasonality
    seasonality = seasonality_strength * np.sin(
        2 * np.pi * t / seasonality_period
    )

    # Noise
    noise = np.random.normal(0, noise_std, size=length)

    # Base signal
    series = trend + seasonality + noise

    # Long-term dependency
    for i in range(long_term_lag, length):
        series[i] += long_term_weight * series[i - long_term_lag]

    return series.reshape(-1, 1)
