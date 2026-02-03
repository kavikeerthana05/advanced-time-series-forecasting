"""
Training loop with early stopping.
"""

import torch
import torch.nn as nn
from torch.optim import Adam

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs=50,
    patience=7
):
    """
    Trains a model with validation and early stopping.

    Returns
    -------
    dict
        Training history containing loss values.
    """
    best_val_loss = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x, y.shape[1])[0] if isinstance(output, tuple) else model(x, y.shape[1])
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                output = model(x, y.shape[1])[0] if isinstance(output, tuple) else model(x, y.shape[1])
                val_loss += criterion(output, y).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    return history
