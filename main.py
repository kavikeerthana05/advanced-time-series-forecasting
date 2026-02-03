"""
Main execution script.
"""
"""
Main entry point for Time Series Forecasting using
LSTM Baseline and Attention-based Seq2Seq model.
"""
"""
Main pipeline for training, evaluating, and comparing
LSTM and Attention-based Seq2Seq models for
multivariate time series forecasting.
"""

import torch
from torch.utils.data import DataLoader

from models.lstm_baseline import LSTMSeq2Seq
from models.attention_seq2seq import AttentionSeq2Seq
from training.train import train_model
from evaluation.metrics import mae,rmse
from evaluation.evaluate import evaluate_model
from visualization.attention_visualization import plot_attention_weights
from datasets.dataset import TimeSeriesDataset


# -------------------- CONFIGURATION --------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
HORIZON = 10

INPUT_SIZE = 5      # number of features
HIDDEN_SIZE = 64
NUM_LAYERS = 2


# -------------------- DATA LOADING --------------------

train_dataset = TimeSeriesDataset(split="train")
val_dataset = TimeSeriesDataset(split="val")
test_dataset = TimeSeriesDataset(split="test")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -------------------- BASELINE LSTM --------------------

lstm_model = LSTMSeq2Seq(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS
).to(DEVICE)

print("\nTraining LSTM Baseline...")
train_model(
    model=lstm_model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=DEVICE,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    horizon=HORIZON
)


# -------------------- ATTENTION MODEL --------------------

attn_model = AttentionSeq2Seq(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS
).to(DEVICE)

print("\nTraining Attention-based Seq2Seq...")
train_model(
    model=attn_model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=DEVICE,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    horizon=HORIZON
)


# -------------------- PHASE 5: MODEL EVALUATION --------------------

lstm_mae, lstm_rmse = evaluate_model(
    model=lstm_model,
    dataloader=test_loader,
    device=DEVICE,
    horizon=HORIZON
)

attn_mae, attn_rmse = evaluate_model(
    model=attn_model,
    dataloader=test_loader,
    device=DEVICE,
    horizon=HORIZON
)

print("\nMODEL COMPARISON RESULTS")
print(f"LSTM Baseline   → MAE: {lstm_mae:.4f}, RMSE: {lstm_rmse:.4f}")
print(f"Attention Model → MAE: {attn_mae:.4f}, RMSE: {attn_rmse:.4f}")


# -------------------- PHASE 4: ATTENTION VISUALIZATION --------------------

x_sample, _ = next(iter(test_loader))
x_sample = x_sample.to(DEVICE)

_, attention_weights = attn_model(x_sample[:1], horizon=HORIZON)

plot_attention_weights(attention_weights)


print("\nPipeline execution completed successfully.")

