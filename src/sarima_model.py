from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_sarima(train, test):
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,24))
    results = model.fit(disp=False)
    forecast = results.forecast(len(test))
    return forecast
