"""
Main execution file for training and evaluating time-series models.
"""

import torch
import optuna
import numpy as np
from torch.utils.data import DataLoader, random_split
from data.dataset import TimeSeriesDataset
from models.attention_seq2seq import AttentionSeq2Seq
from training.train import train_one_epoch, validate
from evaluation.metrics import calculate_metrics
from evaluation.explainability import explain_model_shap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    # Task 3: Automated Hyperparameter Search
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    seq_len = trial.suggest_int("seq_len", 30, 100)
    
    dataset = TimeSeriesDataset('data/multivariate_data.csv', seq_length=seq_len)
    train_size = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = AttentionSeq2Seq(input_dim=3, hidden_dim=hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(10): # Shorter epochs for tuning
        train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
    
    return val_loss

def run_final_training():
    # 1. Hyperparameter Optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    print(f"Optimized Parameters: {best_params}")

    # 2. Final Train with Best Params
    dataset = TimeSeriesDataset('data/multivariate_data.csv', seq_length=best_params['seq_len'])
    train_loader = DataLoader(dataset, batch_size=32)
    
    model = AttentionSeq2Seq(3, best_params['hidden_dim']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    
    # ... [Perform Full Training Loop here] ...

    # 3. Standardized Evaluation (Task 4 & Rec 4)
    model.eval()
    # Logic to get y_true and y_pred...
    # metrics = calculate_metrics(y_true, y_pred)
    
    # 4. Explainability (Task 4)
    explain_model_shap(model, train_loader, DEVICE)

if __name__ == "__main__":
    run_final_training()