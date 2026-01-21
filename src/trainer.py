import torch
import torch.nn as nn

def train_model(model, X, y, epochs=20, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        output = model(X_t)

        # Handle AttentionLSTM (returns output, weights)
        if isinstance(output, tuple):
            preds = output[0]
        else:
            preds = output

        preds = preds.squeeze()
        loss = loss_fn(preds, y_t)

        loss.backward()
        optimizer.step()

    return model