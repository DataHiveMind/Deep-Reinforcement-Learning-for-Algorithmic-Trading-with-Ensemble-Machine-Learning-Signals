import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass

@dataclass
class DataLoader:
    ticker: str
    start_date: str
    end_date: str

    def load_data(self) -> pd.DataFrame:
        """
        Load historical stock data from Yahoo Finance.
        
        Returns:
            pd.DataFrame: DataFrame containing the stock data with 'Date' as index.
        """
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if df is None or df.empty:
            raise ValueError(f"No data found for ticker {self.ticker} between {self.start_date} and {self.end_date}")
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the stock data by filling missing values and calculating returns.

        Args:
            df (pd.DataFrame): DataFrame containing the stock data.
        Returns:
            pd.DataFrame: Preprocessed DataFrame with 'Returns' column.
        """        
        df.ffill(inplace=True)
        df['Returns'] = df['Adj Close'].pct_change()
        df.dropna(inplace=True)
        return df