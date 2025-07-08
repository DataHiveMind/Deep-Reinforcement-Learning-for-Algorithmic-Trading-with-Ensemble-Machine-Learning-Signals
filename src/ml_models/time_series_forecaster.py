import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed, Flatten
from tensorflow.keras import optimizers  

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import coint

from dataclasses import dataclass

@dataclass
class TimeSeriesForecaster:
    """
    A class for time series forecasting using LSTM and ARIMA models.
    """
    
    def __post_init__(self):
        self.lstm_model = None
        self.arima_model = None

    def fit_lstm(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Fit an LSTM model to the training data.
        """
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        self.lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        return self.lstm_model

    def fit_arima(self, y_train):
        """
        Fit an ARIMA model to the training data.
        """
        self.arima_model = sm.tsa.ARIMA(y_train, order=(5, 1, 0))
        self.arima_result = self.arima_model.fit()
        return self.arima_result
    
    def predict_lstm(self, X_test):
        """
        Predict using the fitted LSTM model.
        """
        if self.lstm_model is None:
            raise ValueError("LSTM model has not been fitted yet.")
        return self.lstm_model.predict(X_test)
    
    def predict_arima(self, steps=1):
        """
        Predict using the fitted ARIMA model.
        """
        if not hasattr(self, 'arima_result') or self.arima_result is None:
            raise ValueError("ARIMA model has not been fitted yet.")
        return self.arima_result.get_forecast(steps=steps).predicted_mean
    
    def evaluate_lstm(self, X_test, y_test):
        """
        Evaluate the LSTM model on the test data.
        """
        if self.lstm_model is None:
            raise ValueError("LSTM model has not been fitted yet.")
        predictions = self.predict_lstm(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        return mse
    
    def evaluate_arima(self, y_test):
        """
        Evaluate the ARIMA model on the test data.
        """
        if self.arima_model is None:
            raise ValueError("ARIMA model has not been fitted yet.")
        predictions = self.predict_arima(steps=len(y_test))
        mse = np.mean((predictions - y_test) ** 2)
        return mse
    
    def plot_acf_pacf(self, y):
        """
        Plot the ACF and PACF of the time series.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sm.graphics.tsa.plot_acf(y, lags=40, ax=ax[0])
        sm.graphics.tsa.plot_pacf(y, lags=40, ax=ax[1])
        plt.show()

    def adf_test(self, y):
        """
        Perform the Augmented Dickey-Fuller test to check for stationarity.
        """
        result = adfuller(y)
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        print('Critical Values:')
        critical_values = result[4] if len(result) > 4 else result[2]
        for key, value in critical_values.items():
            print(f'  {key}: {value}')
        return result[1] < 0.05
    
