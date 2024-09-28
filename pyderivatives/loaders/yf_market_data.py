import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


class MarketDataLoader:
    def __init__(self, ticker):

        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

    def get_expiration_dates(self) -> Tuple[str]:
        return self.stock.options

    def load_options_data(self) -> Dict[str, Any]:
        """
        Load the options data (calls and puts) for the available expiration dates.
        """
        options_data = dict()
        yf_stock = self.stock
        expiry_dates = self.get_expiration_dates()
        for expiry_date in expiry_dates:
            options_data_date = yf_stock.option_chain(expiry_date)
            options_data[expiry_date] = options_data_date

        return options_data

    def get_call_data(self, options_data) -> Dict[str, pd.DataFrame]:

        expiry_dates = self.get_expiration_dates()
        calls = [options_data[expiry_date].calls for expiry_date in expiry_dates]
        calls = [call[['strike', 'lastPrice', 'impliedVolatility', 'volume', 'inTheMoney']] for call in calls]
        calls_data = dict(zip(expiry_dates, calls))
        return calls_data

    def get_put_data(self, options_data) -> Dict[str, pd.DataFrame]:

        expiry_dates = self.get_expiration_dates()
        puts = [options_data[expiry_date].puts for expiry_date in expiry_dates]
        puts = [put[['strike', 'lastPrice', 'impliedVolatility', 'volume', 'inTheMoney']] for put in puts]
        puts_data = dict(zip(expiry_dates, puts))

        return puts_data

    def get_spot_price(self) -> float:
        """
        Get the current spot price (last traded price) of the stock.
        """
        return self.stock.history(period='1d')['Close'].iloc[-1]

    def get_stock_dividend_level(self, spot_price: float) -> float:
        dividends = self.stock.dividends
        # Filter dividends from the last 12 months
        last_12_months_dividends = dividends[dividends.index > (dividends.index[-1] - pd.DateOffset(years=1))]
        # Calculate the annual dividend (sum of dividends from the last 12 months)
        annual_dividend = last_12_months_dividends.sum()
        # Calculate the dividend yield
        dividend_yield = (annual_dividend / spot_price) * 100

        return dividend_yield

    def get_historical_data(self, period='max', interval='1d'):
        return self.stock.history(period=period, interval=interval)

    @staticmethod
    def get_historical_volatility(stock_data: pd.DataFrame, number_years: float) -> float:

        stock_data = stock_data.tail(int(number_years*252)).copy()
        # Calculate daily returns
        stock_data['Returns'] = stock_data['Close'].pct_change()

        daily_volatility = np.std(stock_data['Returns'].dropna())
        annualized_volatility = daily_volatility * np.sqrt(252)

        return annualized_volatility
