import pandas as pd 
import numpy as np
import statsmodels.api as sm
from dataclasses import dataclass

@dataclass
class FeatureEngineer:
    """
    Class for feature engineering on stock data.
    """
    
    def __post_init__(self):
        self.features = pd.DataFrame()

    def add_moving_average(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Add moving average feature to the data.
        """
        self.features[f'ma_{window}'] = data['close'].rolling(window=window).mean()
        return self.features

    def add_bollinger_bands(self, data: pd.DataFrame, window: int, num_std: int) -> pd.DataFrame:
        """
        Add Bollinger Bands features to the data.
        """
        ma = data['close'].rolling(window=window).mean()
        std = data['close'].rolling(window=window).std()
        self.features[f'bb_upper_{window}'] = ma + (num_std * std)
        self.features[f'bb_lower_{window}'] = ma - (num_std * std)
        return self.features

    def add_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add MACD feature to the data.
        """
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        self.features['macd'] = macd
        self.features['macd_signal'] = signal
        return self.features

    def add_rsi(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Add RSI feature to the data.
        """
        delta = data['close'].diff(1).astype(float)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        self.features[f'rsi_{window}'] = rsi
        return self.features
    
