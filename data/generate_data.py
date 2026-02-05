import numpy as np
import pandas as pd
import os

def generate_multivariate_series(n_steps=10000):
    os.makedirs('data', exist_ok=True)
    time = np.linspace(0, 500, n_steps)
    
    # Target: Seasonality + Trend + Sine Waves
    target = np.sin(time) + 0.5 * np.cos(time * 0.5) + 0.01 * time + np.random.normal(0, 0.05, n_steps)
    # Exogenous features that correlate with target
    feat1 = np.cos(time) + np.random.normal(0, 0.02, n_steps)
    feat2 = np.sin(time * 0.2) + np.random.normal(0, 0.02, n_steps)
    
    df = pd.DataFrame({'target': target, 'feat1': feat1, 'feat2': feat2})
    df.to_csv('data/multivariate_data.csv', index=False)
    print("Dataset created: data/multivariate_data.csv")

if __name__ == "__main__":
    generate_multivariate_series()