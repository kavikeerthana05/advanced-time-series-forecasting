"""
Main execution file for training and evaluating time-series models.
"""

import torch
from torch.utils.data import DataLoader, random_split

from data.generate_data import generate_synthetic_series
from data.dataset import TimeSeriesDataset
from models.lstm_baseline import LSTMBaseline
from models.attention_seq2seq import AttentionSeq2Seq
from training.training import train_model
from evaluation.evaluate import evaluate_model
from visualization.attention_visualization import plot_attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Generate data
series = generate_synthetic_series()

dataset = TimeSeriesDataset(series, input_window=30, forecast_horizon=10)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

# 2. Models
lstm_model = LSTMBaseline(1, 64, 1).to(device)
attention_model = AttentionSeq2Seq(1, 64, 1).to(device)

criterion = torch.nn.MSELoss()

# 3. Train
train_model(lstm_model, train_loader, val_loader, criterion, device)
train_model(attention_model, train_loader, val_loader, criterion, device)

# 4. Evaluate
lstm_metrics = evaluate_model(lstm_model, test_loader, device, horizon=10)
attention_metrics = evaluate_model(attention_model, test_loader, device, horizon=10)

print("LSTM:", lstm_metrics)
print("Attention:", attention_metrics)

# 5. Attention visualization
plot_attention(attention_model, test_loader, device)
