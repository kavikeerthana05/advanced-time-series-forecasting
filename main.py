"""
Main execution script.
"""
import pandas as pd
from torch.utils.data import DataLoader
from datasets.dataset import TimeSeriesDataset
from models.attention_seq2seq import AttentionSeq2Seq
from training.training import train_model
from evaluation.metrics import mae, rmse

df = pd.read_csv("data/time_series.csv")

dataset = TimeSeriesDataset(df)
loader = DataLoader(dataset, batch_size=32)

model = AttentionSeq2Seq(
    input_dim=df.shape[1],
    hidden_dim=64,
    num_layers=2,
    num_heads=4
)

for x, y in loader:
    preds, attn = model(x, y.shape[1])
    print(preds.shape)
    break
