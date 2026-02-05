import torch
import optuna

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.squeeze(), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output.squeeze(), y)
            total_loss += loss.item()
    return total_loss / len(loader)

def objective(trial, train_loader, test_loader, device, input_dim):
    # Search Space
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    from models.attention_seq2seq import AttentionSeq2Seq
    model = AttentionSeq2Seq(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(10): # Short tuning epochs
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, test_loader, criterion, device)
        
    return val_loss