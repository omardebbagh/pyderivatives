import pandas as pd
from typing import List, Dict, Union
from scipy.optimize import minimize
from pyderivatives.utils import *
from pyderivatives.loaders.yf_market_data import MarketDataLoader


class SABR:
    def __init__(self, alpha=None, beta=1, rho=None, nu=None):
        """
        Parameters:
        If None, will need to calibrate.
        """

        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

    def sabr_vol(self, forward: float, strike: float, maturity: float) -> float:
        """
        SABR volatility closed formula with Hagan's approximation.
        """
        fwd, stk, mat = forward, strike, maturity
        if fwd == stk:
            # ATM formula
            vol = (self.alpha / (fwd ** (1 - self.beta))) * (
                        1 + ((1 - self.beta) ** 2 / 24 * (self.alpha ** 2 / (fwd ** (2 - 2 * self.beta)))) +
                        (self.rho * self.beta * self.nu * self.alpha / (4 * fwd ** (1 - self.beta))) +
                        ((2 - 3 * self.rho ** 2) * self.nu ** 2 / 24) * mat)
        else:
            z = (self.nu / self.alpha) * ((fwd * stk) ** ((1 - self.beta) / 2)) * np.log(fwd / stk)
            x_z = np.log((np.sqrt(1 - 2 * self.rho * z + z ** 2) + z - self.rho) / (1 - self.rho))
            vol = (self.alpha / (
                        (fwd * stk) ** ((1 - self.beta) / 2) * (1 + ((1 - self.beta) ** 2 / 24 * np.log(fwd / stk) ** 2)))) * (
                              z / x_z)

        return vol

    @staticmethod
    def objective_function(sabr_params: List[float], market_vols: List[float],
                           strikes: List[int], forward: float, maturity: float) -> float:
        """
        Objective function for SABR calibration.

        Parameters:
        sabr_params: [alpha, beta, rho, nu].
        market_vols: Market implied volatilities.
        strikes: Strike prices corresponding to market volatilities.

        Returns:
        Sum of squared errors between SABR volatilities and market volatilities.
        """
        alpha, beta, rho, nu = sabr_params
        sabr_model = SABR(alpha, beta, rho, nu)
        sabr_vols = [sabr_model.sabr_vol(forward, strike, maturity) for strike in strikes]
        return np.sum((np.array(sabr_vols) - np.array(market_vols)) ** 2)

    @staticmethod
    def calibrate_skew(ticker: str, forward: float, maturity: float, risk_free_rate: float,
                       dividend_yield, initial_params=(0.2, 1, 0, 0.3)) -> Dict[str, Union[list, float]]:
        """
        Calibrate the SABR model to market implied volatilities.
        """
        # load market data
        market_data_loader = MarketDataLoader(ticker=ticker)
        spot = market_data_loader.get_spot_price()
        options_data = market_data_loader.load_options_data()
        calls_data_all = market_data_loader.get_call_data(options_data=options_data)
        puts_data_all = market_data_loader.get_put_data(options_data=options_data)
        options_matu_dates = market_data_loader.get_expiration_dates()

        # choosing the closest maturity available if exact maturity date not found
        if maturity_to_date(maturity) not in options_matu_dates:
            matu_date = closest_date(maturity_years=maturity, available_dates=options_matu_dates)
            matu_years = date_to_maturity(matu_date)
            fwd = get_forward(spot=spot,
                              dividend_yield=dividend_yield,
                              maturity=matu_years,
                              risk_free_rate=risk_free_rate)
        else:
            matu_date = maturity_to_date(maturity)
            matu_years = maturity
            fwd = forward

        calls_data = calls_data_all[matu_date]
        puts_data = puts_data_all[matu_date]
        
        market_vols = pd.merge(calls_data[['strike', 'impliedVolatility']],
                               puts_data[['strike', 'impliedVolatility']],
                               on='strike', how='outer', suffixes=('_call', '_put'))

        market_vols['impliedVolatility_call'] = market_vols['impliedVolatility_call'].fillna(0)
        market_vols['impliedVolatility_put'] = market_vols['impliedVolatility_put'].fillna(0)

        # calls IV should be same as puts IV in theory for same strike so considered average of two if different
        market_vols['IV'] = market_vols[['impliedVolatility_call', 'impliedVolatility_put']].mean(axis=1)
        market_vols = market_vols[['strike', 'IV']]
        market_vols = market_vols.sort_values(by='strike', ascending=True).reset_index(drop=True)

        result = minimize(SABR.objective_function, initial_params, args=(list(market_vols['IV']),
                                                                         list(market_vols['strike']),
                                                                         fwd, matu_years),
                          bounds=[(0.001, None), (1, 1), (-1, 1), (0, None)])
        calibration_results = {'calibrated_params': list(result.x),
                               'maturity_calibration': matu_years}
        return calibration_results

