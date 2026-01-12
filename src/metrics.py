import numpy as np

def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))

def mae(y, yhat):
    return np.mean(np.abs(y - yhat))

def mape(y, yhat):
    return np.mean(np.abs((y - yhat) / y)) * 100
