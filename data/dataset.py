import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_file, seq_length=50, target_col='target'):
        df = pd.read_csv(csv_file)
        self.scaler = StandardScaler()
        # Scale all columns
        data = self.scaler.fit_transform(df.values)
        
        self.X, self.y = [], []
        for i in range(len(data) - seq_length):
            self.X.append(data[i:i+seq_length, :]) # All features
            self.y.append(data[i+seq_length, 0])   # Target (column 0)
            
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]