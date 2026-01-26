"""
Training loop with early stopping.
"""

import torch
import torch.nn as nn
from torch.optim import Adam

def train_model(model, train_loader, val_loader, epochs=30):
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x, y.shape[1])
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")
