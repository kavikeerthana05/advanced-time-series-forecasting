import numpy as np
import pandas as pd

def generate_data(
    n_series=5,
    n_steps=5000,
    seasonal_period=24,
    seed=42
):
    np.random.seed(seed)
    time = np.arange(n_steps)

    data = []
    for i in range(n_series):
        trend = 0.0005 * time
        seasonal = np.sin(2 * np.pi * time / seasonal_period)
        long_term = np.cumsum(np.random.normal(0, 0.02, n_steps))
        noise = np.random.normal(0, 0.3, n_steps)

        series = trend + seasonal + long_term + noise
        data.append(series)

    df = pd.DataFrame(np.array(data).T,
                      columns=[f"var_{i}" for i in range(n_series)])
    df.to_csv("data/time_series.csv", index=False)
    print("Dataset saved to data/time_series.csv")

if __name__ == "__main__":
    generate_data()
