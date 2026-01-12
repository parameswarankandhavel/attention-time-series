import torch
import torch.nn as nn

def train_model(model, X, y, epochs=20, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t).squeeze()
        loss = loss_fn(preds, y_t)
        loss.backward()
        optimizer.step()

    return model
