import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """
    Sliding window dataset.
    """

    def __init__(self, data, input_len=96, horizon=24):
        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.input_len = input_len
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.input_len - self.horizon

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.input_len]
        y = self.data[idx+self.input_len:idx+self.input_len+self.horizon]
        return x, y
