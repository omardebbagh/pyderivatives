from datetime import datetime, timedelta
import numpy as np
from typing import Tuple


def maturity_to_date(maturity_years: float):
    # Assuming start date is today
    start_date = datetime.today()
    maturity_delta = timedelta(days=maturity_years * 365)
    maturity_date = start_date + maturity_delta
    return maturity_date.strftime('%Y-%m-%d')


def closest_date(maturity_years: float, available_dates: Tuple[str]):
    # Convert maturity years to target date
    target_date_str = maturity_to_date(maturity_years)
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')

    # Convert available dates from strings to datetime objects
    available_dates = [datetime.strptime(date, '%Y-%m-%d') for date in available_dates]

    # Find the closest date
    closest_date = min(available_dates, key=lambda d: abs(d - target_date))

    return closest_date.strftime('%Y-%m-%d')


def date_to_maturity(date_str):
    # Convert the date string to a datetime object
    target_date = datetime.strptime(date_str, '%Y-%m-%d')

    # Assuming the reference date is today
    start_date = datetime.today()

    # Calculate the difference in days
    delta_days = (target_date - start_date).days

    # Convert days to years
    maturity_years = delta_days / 365

    return round(maturity_years, 2)


def get_forward(spot, risk_free_rate, dividend_yield, maturity):
    return spot * np.exp((risk_free_rate-dividend_yield)) * maturity
