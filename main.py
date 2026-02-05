import torch
import numpy as np
import pandas as pd
import optuna
from captum.attr import IntegratedGradients
from data.dataset import TimeSeriesDataset
from models.attention_seq2seq import AttentionSeq2Seq
from models.lstm_baseline import LSTMBaseline
from training.train import train_one_epoch, validate, objective
from evaluation.metrics import calculate_metrics
from statsmodels.tsa.statespace.sarimax import SARIMAX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_sarima_baseline():
    print("\n--- Running SARIMA Baseline ---")
    df = pd.read_csv('data/multivariate_data.csv')
    train_data = df['target'].values[:800]
    test_data = df['target'].values[800:1000]
    
    model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    res = model.fit(disp=False)
    preds = res.forecast(steps=len(test_data))
    
    metrics = calculate_metrics(test_data, preds)
    print(f"SARIMA Results: {metrics}")

def explain_model(model, test_loader):
    print("\n--- Generating Explainability Analysis (Task 4) ---")
    ig = IntegratedGradients(model)
    x, _ = next(iter(test_loader))
    x = x[:1].to(DEVICE).requires_grad_()
    
    attributions = ig.attribute(x, target=0)
    importance = attributions.abs().mean(dim=1).squeeze().detach().cpu().numpy()
    print(f"Feature Importance (Integrated Gradients) for first sample: {importance}")

if __name__ == "__main__":
    # 1. Generate Data
    import data.generate_data as gd
    gd.generate_multivariate_series()
    
    # 2. SARIMA Baseline (Deliverable 2 Requirement)
    run_sarima_baseline()
    
    # 3. Hyperparameter Tuning (Task 3)
    dataset = TimeSeriesDataset('data/multivariate_data.csv')
    train_ds, test_ds = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, test_loader, DEVICE, 3), n_trials=5)
    
    print(f"Best Hyperparameters: {study.best_params}")
    
    # 4. Final Train with Best Params
    best_model = AttentionSeq2Seq(3, study.best_params['hidden_dim']).to(DEVICE)
    # ... (Run standard training loop here) ...
    
    # 5. Explainability
    explain_model(best_model, test_loader)