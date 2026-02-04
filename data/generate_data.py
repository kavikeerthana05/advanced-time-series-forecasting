import numpy as np
import pandas as pd

def generate_synthetic_series(
    n_series: int = 5,
    n_steps: int = 5000,
    seasonal_period: int = 24,
    noise_std: float = 0.3,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a multivariate time series with seasonality and long-term dependencies.

    Returns
    -------
    pd.DataFrame
        Shape: (n_steps, n_series)
    """
    np.random.seed(seed)
    time = np.arange(n_steps)

    data = []
    for i in range(n_series):
        trend = 0.0005 * time
        seasonal = np.sin(2 * np.pi * time / seasonal_period)
        long_memory = np.cumsum(np.random.normal(0, 0.02, n_steps))
        noise = np.random.normal(0, noise_std, n_steps)

        series = trend + seasonal + long_memory + noise
        data.append(series)

    return pd.DataFrame(np.array(data).T, columns=[f"var_{i}" for i in range(n_series)])
