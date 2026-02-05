"""
Main execution file for training and evaluating time-series models.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from data.dataset import TimeSeriesDataset
from models.attention_seq2seq import AttentionSeq2Seq
from models.lstm_baseline import LSTMBaseline
from training.train import train_one_epoch, validate
from evaluation.metrics import calculate_metrics
import os

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 30
SEQ_LEN = 50

def run_experiment(model_type='attention'):
    dataset = TimeSeriesDataset('data/multivariate_data.csv', seq_length=SEQ_LEN)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    if model_type == 'attention':
        model = AttentionSeq2Seq(3, 64).to(DEVICE)
    else:
        model = LSTMBaseline(3, 64).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    print(f"--- Training {model_type} ---")
    for epoch in range(EPOCHS):
        t_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        v_loss = validate(model, test_loader, criterion, DEVICE)
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Val Loss: {v_loss:.4f}")

    # Final Eval
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out, _ = model(x.to(DEVICE))
            preds.extend(out.squeeze().cpu().numpy())
            actuals.extend(y.numpy())
    
    metrics = calculate_metrics(np.array(actuals), np.array(preds))
    print(f"Results for {model_type}: {metrics}")

if __name__ == "__main__":
    if not os.path.exists('data/multivariate_data.csv'):
        import data.generate_data as gd
        gd.generate_multivariate_series()
    
    run_experiment('baseline')
    run_experiment('attention')