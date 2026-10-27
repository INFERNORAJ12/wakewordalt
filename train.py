import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_network(
    model: nn.Module,
    data_loader: DataLoader,
    epochs: int = 8,
    lr: float = 0.001
) -> nn.Module:
    """
    Executes binary cross-entropy optimization across batches.
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_x, batch_y in data_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        if epoch % 2 == 0 or epoch == epochs:
            print(f"  [Train] Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f}")

    return model