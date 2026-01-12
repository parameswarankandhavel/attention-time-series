import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # predict load
    return np.array(X), np.array(y)

def preprocess(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler
