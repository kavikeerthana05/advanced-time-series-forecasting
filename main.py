import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from data.dataset import TimeSeriesDataset
from models.attention_seq2seq import AttentionSeq2Seq
from training.train import train_one_epoch
from evaluation.metrics import calculate_metrics
import data.generate_data as gd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_final_experiment():
    gd.generate_multivariate_series()
    dataset = TimeSeriesDataset('data/multivariate_data.csv')
    
    train_size = int(0.8 * len(dataset))
    train_ds, test_ds = random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    # Input_dim is now 6 (target + 5 features)
    model = AttentionSeq2Seq(input_dim=6, hidden_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("--- Training Final Model with Quantile Regression ---")
    for epoch in range(20):
        loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        if epoch % 5 == 0: print(f"Epoch {epoch} Loss: {loss:.4f}")

    # Evaluation
    model.eval()
    all_preds, all_actuals = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out, _ = model(x.to(DEVICE))
            all_preds.append(out.cpu().numpy())
            all_actuals.append(y.numpy())

    preds = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)
    
    # Use index 1 (0.5 quantile) for point forecast metrics (MAE/RMSE)
    metrics = calculate_metrics(actuals, preds[:, 1], y_train=actuals, quantiles=preds)
    print("\n--- Final Performance Summary Table ---")
    print(pd.DataFrame([metrics]).to_markdown())
# ... (rest of your main.py code remains the same)

    # Use index 1 (0.5 quantile) for point forecast metrics (MAE/RMSE)
    metrics = calculate_metrics(actuals, preds[:, 1], y_train=actuals, quantiles=preds)
    
    print("\n--- Final Performance Summary Table ---")
    try:
        # Attempt to print as a nice markdown table
        print(pd.DataFrame([metrics]).to_markdown(index=False))
    except ImportError:
        # Fallback to standard print if tabulate is missing
        print(pd.DataFrame([metrics]).to_string(index=False))

if __name__ == "__main__":
    run_final_experiment()
