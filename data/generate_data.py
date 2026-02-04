import numpy as np

def generate_synthetic_series(length=2000):
    t = np.arange(length)
    trend = 0.01 * t
    seasonality = np.sin(2 * np.pi * t / 50)
    noise = np.random.normal(0, 0.1, size=length)

    series = trend + seasonality + noise

    for i in range(25, length):
        series[i] += 0.5 * series[i - 25]

    return series.reshape(-1, 1)
