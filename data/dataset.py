import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, series, input_window, forecast_horizon):
        self.series = torch.tensor(series, dtype=torch.float32)
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.series) - self.input_window - self.forecast_horizon

    def __getitem__(self, idx):
        x = self.series[idx:idx + self.input_window]
        y = self.series[idx + self.input_window:
                        idx + self.input_window + self.forecast_horizon]
        return x, y
