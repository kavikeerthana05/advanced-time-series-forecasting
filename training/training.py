"""
Training loop with early stopping.
"""

import torch
import torch.nn as nn
from torch.optim import Adam

def train_model(model, train_loader, val_loader, criterion, device, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

