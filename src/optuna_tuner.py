import optuna
import torch
from src.attention_lstm import AttentionLSTM
from src.trainer import train_model

def objective(trial, X_train, y_train, input_size):
    hidden_size = trial.suggest_int("hidden_size", 32, 128)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    epochs = trial.suggest_int("epochs", 10, 40)

    model = AttentionLSTM(input_size, hidden_size)
    model = train_model(model, X_train, y_train, epochs=epochs, lr=lr)

    with torch.no_grad():
        preds, _ = model(torch.tensor(X_train, dtype=torch.float32))
        loss = torch.mean((preds.squeeze() - torch.tensor(y_train, dtype=torch.float32)) ** 2)

    return loss.item()

def tune(X_train, y_train, input_size):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, input_size), n_trials=10)

    best_params = study.best_params
    model = AttentionLSTM(input_size, best_params["hidden_size"])
    model = train_model(model, X_train, y_train, best_params["epochs"], best_params["lr"])

    return model, best_params
