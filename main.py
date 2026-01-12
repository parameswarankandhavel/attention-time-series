import pandas as pd
import torch

from src.preprocessing import preprocess, create_sequences
from src.sarima_model import train_sarima
from src.lstm_model import LSTM
from src.attention_lstm import AttentionLSTM
from src.trainer import train_model
from src.metrics import rmse, mae, mape
from src.optuna_tuner import tune

# Load data
df = pd.read_csv("data/synthetic_timeseries.csv")
scaled, _ = preprocess(df)

X, y = create_sequences(scaled)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# SARIMA
sarima_pred = train_sarima(df["load"][:split], df["load"][split:])
print("\nSARIMA RMSE:", rmse(df["load"][split:].values, sarima_pred.values))

# Standard LSTM
lstm = LSTM(X.shape[2])
lstm = train_model(lstm, X_train, y_train)
with torch.no_grad():
    lstm_preds = lstm(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()

print("LSTM RMSE:", rmse(y_test, lstm_preds))

# Attention + Optuna
att_model, best_params = tune(X_train, y_train, X.shape[2])

with torch.no_grad():
    att_preds, att_weights = att_model(torch.tensor(X_test, dtype=torch.float32))

att_preds = att_preds.squeeze().numpy()

print("\nAttention Model Metrics")
print("RMSE:", rmse(y_test, att_preds))
print("MAE:", mae(y_test, att_preds))
print("MAPE:", mape(y_test, att_preds))

print("\nBest Attention Hyperparameters:", best_params)

# Attention interpretation
mean_attention = att_weights.mean(dim=0).squeeze().numpy()
top_steps = mean_attention.argsort()[-3:][::-1]

print("\nMost important time steps (attention):")
for t in top_steps:
    print(f"t-{len(mean_attention)-t} with weight {mean_attention[t]:.4f}")
