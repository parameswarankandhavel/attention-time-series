import pandas as pd
import torch

from src.preprocessing import preprocess, create_sequences
from src.sarima_model import train_sarima
from src.lstm_model import LSTM
from src.attention_lstm import AttentionLSTM
from src.trainer import train_model
from src.metrics import rmse, mae, mape
from src.optuna_tuner import tune


def main():
    # --------------------------------------------------
    # 1. Load dataset (already generated)
    # --------------------------------------------------
    df = pd.read_csv("data/synthetic_timeseries.csv")

    # --------------------------------------------------
    # 2. Preprocessing
    # --------------------------------------------------
    scaled_data, _ = preprocess(df)
    X, y = create_sequences(scaled_data, seq_length=24)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --------------------------------------------------
    # 3. SARIMA Baseline
    # --------------------------------------------------
    sarima_pred = train_sarima(
        df["load"][:split],
        df["load"][split:]
    )

    print("\nSARIMA MODEL")
    print("RMSE:", rmse(df["load"][split:].values, sarima_pred.values))

    # --------------------------------------------------
    # 4. Standard LSTM
    # --------------------------------------------------
    lstm = LSTM(input_size=X.shape[2])
    lstm = train_model(lstm, X_train, y_train)

    with torch.no_grad():
        lstm_preds = lstm(
            torch.tensor(X_test, dtype=torch.float32)
        ).squeeze().numpy()

    print("\nSTANDARD LSTM MODEL")
    print("RMSE:", rmse(y_test, lstm_preds))

    # --------------------------------------------------
    # 5. Attention LSTM + Optuna (Bayesian Optimization)
    # --------------------------------------------------
    att_model, best_params = tune(
        X_train, y_train, input_size=X.shape[2]
    )

    with torch.no_grad():
        att_preds, att_weights = att_model(
            torch.tensor(X_test, dtype=torch.float32)
        )

    att_preds = att_preds.squeeze().numpy()

    print("\nATTENTION LSTM MODEL")
    print("RMSE:", rmse(y_test, att_preds))
    print("MAE:", mae(y_test, att_preds))
    print("MAPE:", mape(y_test, att_preds))

    print("\nBest Attention Hyperparameters:")
    print(best_params)

    # --------------------------------------------------
    # 6. Attention Interpretation
    # --------------------------------------------------
    mean_attention = att_weights.mean(dim=0).squeeze().numpy()
    top_steps = mean_attention.argsort()[-3:][::-1]

    print("\nMost important time steps (attention):")
    for t in top_steps:
        print(f"t-{len(mean_attention) - t} with weight {mean_attention[t]:.4f}")


if __name__ == "__main__":
    main()
