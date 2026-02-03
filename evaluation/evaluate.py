import torch
from evaluation.metrics import mae, rmse

def evaluate_model(model, dataloader, device, horizon):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            output = model(x, horizon=horizon)
            preds.append(output.cpu())
            targets.append(y[:, :horizon, :].cpu())

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    return {
        "MAE": mae(preds, targets),
        "RMSE": rmse(preds, targets)
    }
