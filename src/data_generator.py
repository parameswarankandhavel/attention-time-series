import numpy as np
import pandas as pd

def generate_dataset(n_points=6000, seed=42):
    np.random.seed(seed)
    time = np.arange(n_points)

    daily = np.sin(2 * np.pi * time / 24)
    weekly = np.sin(2 * np.pi * time / (24 * 7))

    trend = time * 0.0005

    noise = np.random.normal(0, 1 + 0.5 * np.sin(time / 200), n_points)

    load = 50 + 10 * daily + 5 * weekly + trend + noise
    temperature = 20 + 10 * daily + np.random.normal(0, 2, n_points)
    humidity = 60 + 10 * weekly + np.random.normal(0, 5, n_points)
    wind = 5 + np.random.normal(0, 1, n_points)
    pressure = 1013 + np.random.normal(0, 2, n_points)

    df = pd.DataFrame({
        "load": load,
        "temperature": temperature,
        "humidity": humidity,
        "wind": wind,
        "pressure": pressure
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/synthetic_timeseries.csv", index=False)
