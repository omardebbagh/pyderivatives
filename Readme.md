# pyderivatives - Derivatives Pricing with Black-Scholes, Monte Carlo (MC), LS-MC, and SABR Volatility Model

## Project Overview

This project focuses on pricing financial derivatives using several models, including:
- **Black-Scholes Model**: For pricing European options.
- **Monte Carlo Simulation (MC)**: Useful for pricing complex derivatives.
- **Longstaff-Schwartz Monte Carlo (LS-MC)**: Specifically for American style derivatives.
- **SABR Volatility Model**: To incorporate volatility skew and vol of vol for more complex payoffs.

## Features

- You can either give the vanilla pricer fixed parameters: spot level, dividend yield, volatility 
 that are necessary or put simply a ticker with a strike, maturity and rf rate.
- When pricing a ticker instead of fixed parameters, 
 data retrieval is done via yFinance API: Automatically fetches relevant market data (such as historical volatility, dividends, spot level) for the specified ticker to use in pricing models.
- Note: Due to limitations of the yFinance API, it may not provide complete data for some tickers, particularly when attempting to calibrate or use the SABR model due to insufficient or non-existent options quotes.
- Tickers that I used to test my code and seem to have enough data are for example 'AAPL', 'AMZN'

## To Do

- Improve data reliability by integrating fallback APIs when yFinance lacks sufficient data.
- Add more complex derivatives payoffs where SABR volatility model is more relevant to be used. 

## Dependencies

- Python 3.x
- yFinance API
- Numpy, Scipy, Pandas




