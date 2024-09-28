import warnings
import numpy as np
from scipy.stats import norm
from typing import Union, Optional, List, Dict, Any
from pyderivatives.loaders.yf_market_data import MarketDataLoader
from pyderivatives.volatility_models.sabr_model import SABR
from pyderivatives.utils import *


class VanillaOption:
    def __init__(self, udl_spot: Union[str, float], strike: float, maturity: float,
                 risk_free_rate: float, sigma: Optional[float] = None, div_yield: Optional[float] = None,
                 option_type='call', option_style='EUR'):

        self._udl_spot = udl_spot
        self.strike = strike
        self.maturity = maturity  # maturity in years
        self.risk_free_rate = risk_free_rate
        self._sigma = sigma
        self._div_yield = div_yield
        self.option_type = option_type
        self.option_style = option_style

    @property
    def spot(self):
        if isinstance(self._udl_spot, str):
            market_loader = MarketDataLoader(self._udl_spot)
            spot = market_loader.get_spot_price()
        else:
            spot = self._udl_spot
        return spot

    @property
    def volatility(self):
        if self._sigma:
            return self._sigma
        elif isinstance(self._udl_spot, str):
            market_loader = MarketDataLoader(self._udl_spot)
            stocks_data = market_loader.get_historical_data()
            volatility = MarketDataLoader.get_historical_volatility(stocks_data,
                                                                    self.maturity)
            return volatility
        else:
            raise ValueError("Insert Volatility level")

    @property
    def dividend_yield(self):
        if self._div_yield:
            return self._div_yield
        elif isinstance(self._udl_spot, str) and self._div_yield is None:
            market_loader = MarketDataLoader(self._udl_spot)
            dividend_yield = market_loader.get_stock_dividend_level(self.spot)
            return dividend_yield/100
        else:
            warnings.warn("Dividends not taken into account")
            return 0

    def _black_scholes_price(self, volatility: float) -> float:
        spot, strike, maturity, rf, div_yield = self.spot, self.strike, self.maturity, self.risk_free_rate, self.dividend_yield

        d1 = (np.log(spot / strike) + (rf - div_yield + 0.5 * volatility ** 2) * maturity) / (
                    volatility * np.sqrt(maturity))
        d2 = d1 - volatility * np.sqrt(maturity)

        if self.option_type == 'call':
            # Call option price
            price = (spot * np.exp(-div_yield * maturity) * norm.cdf(d1) - strike * np.exp(-rf * maturity) * norm.cdf(
                d2))
        elif self.option_type == 'put':
            # Put option price
            price = (strike * np.exp(-rf * maturity) * norm.cdf(-d2) - spot * np.exp(-div_yield * maturity) * norm.cdf(
                -d1))
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")

        return price

    def _mc_price(self, num_simulations=1000, use_antithetic=True) -> float:

        # Total number of time steps based on maturity in years and 252 trading days per year
        num_steps = int(self.maturity * 252)

        np.random.seed(0)
        dt = self.maturity / num_steps  # Time step size

        discount_factor = np.exp(-(self.risk_free_rate - self.dividend_yield) * self.maturity)

        # Simulate paths
        z = np.random.normal(size=(num_simulations, num_steps))

        drift = (self.risk_free_rate - self.dividend_yield - 0.5 * self.volatility ** 2) * dt
        diffusion = self.volatility * np.sqrt(dt)

        # Initialize matrices for spot price paths (each row is one path)
        spot_paths = np.zeros((num_simulations, num_steps + 1))
        spot_paths[:, 0] = self.spot  # Set initial spot price

        # Simulate the antithetic path
        if use_antithetic:
            z_antithetic = -z

            # Initialize a matrix for antithetic spot paths
            spot_paths_antithetic = np.zeros((num_simulations, num_steps + 1))
            spot_paths_antithetic[:, 0] = self.spot  # Set initial spot price

        # Simulate the paths step by step
        for t in range(1, num_steps + 1):
            spot_paths[:, t] = spot_paths[:, t - 1] * np.exp(drift + diffusion * z[:, t - 1])

            if use_antithetic:
                spot_paths_antithetic[:, t] = spot_paths_antithetic[:, t - 1] * np.exp(
                    drift + diffusion * z_antithetic[:, t - 1])

        # Calculate payoffs at maturity
        final_spots = spot_paths[:, -1]
        payoffs = np.maximum(self.strike - final_spots, 0) if self.option_type == 'put' else np.maximum(
            final_spots - self.strike, 0)

        if use_antithetic:
            final_spots_antithetic = spot_paths_antithetic[:, -1]
            payoffs_antithetic = np.maximum(self.strike - final_spots_antithetic, 0) if self.option_type == 'put' else np.maximum(
                final_spots_antithetic - self.strike, 0)

            # Average the payoffs from both the original and antithetic paths
            payoffs = 0.5 * (payoffs + payoffs_antithetic)

        # Discount and average payoffs
        return discount_factor * np.mean(payoffs)

    def _longstaff_mc_price(self, volatility, num_steps: Optional[int] = None, num_paths=10000) -> float:

        # num_steps left variable to handle (weekly,quarterly,semi-annual...) type of US payoffs
        if num_steps is None:
            # The total number of time steps where option can be exercised is daily by default
            num_steps = int(self.maturity * 252)

        dt = self.maturity / num_steps  # Time step size
        discount_factor = np.exp(-self.risk_free_rate * dt)  # Discount factor per time step

        # Simulate the asset price paths with rows as paths and columns as time steps
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.spot
        for time_step in range(1, num_steps + 1):
            z = np.random.standard_normal(num_paths)
            paths[:, time_step] = paths[:, time_step - 1] * np.exp((self.risk_free_rate - self.dividend_yield - 0.5 * volatility ** 2) * dt
                                                   + volatility * np.sqrt(dt) * z)

        # Calculate the payoff of the option at each time step
        if self.option_type == 'call':
            payoffs = np.maximum(paths - self.strike, 0)
        elif self.option_type == 'put':
            payoffs = np.maximum(self.strike - paths, 0)
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")

        # Initialize the optimal cash flow matrix with zeros
        cash_flows = np.zeros((num_paths, num_steps + 1))

        # Backward recursive process to fill the cash flow matrix
        option_values = payoffs[:, -1]  # Initialize option values at maturity
        for time_step in range(num_steps - 1, 0, -1):  # Iterate backward from the second last time step to the first
            # Only consider paths that are in-the-money
            in_the_money = payoffs[:, time_step] > 0
            if np.sum(in_the_money) == 0:
                continue  # Skip if no paths are in-the-money

            # Fit a least-squares polynomial regression to estimate continuation value
            x = paths[in_the_money, time_step]  # Stock prices at time step where in the money
            y = option_values[in_the_money] * discount_factor  # Discounted cash flows at time step + 1
            A = np.vstack([np.ones_like(x), x, x ** 2]).T
            coeffs = np.linalg.lstsq(A, y, rcond=None)[0]  # Polynomial coefficients

            # Calculate the estimated continuation value
            continuation_values = coeffs[0] + coeffs[1] * x + coeffs[2] * x ** 2

            # Determine whether to exercise the option or not
            exercise_values = payoffs[in_the_money, time_step]
            exercise_decision = exercise_values > continuation_values

            # Update cash flows matrix based on the exercise decision
            cash_flows[in_the_money, time_step] = np.where(exercise_decision,
                                                           exercise_values * discount_factor**time_step, 0)
            cash_flows[in_the_money, time_step + 1] = np.where(~exercise_decision,
                                                               option_values[in_the_money] * discount_factor ** (time_step+ 1),
                                                               0)

            # Update option values to use for next time step regression
            option_values = payoffs[:, time_step]

            # Ensure no future cash flows in paths where option is exercised
            exercised_paths = np.where(in_the_money)[0][exercise_decision]  # Indices of exercised paths
            for idx in exercised_paths:
                cash_flows[idx, time_step + 1:] = 0

        return np.sum(cash_flows)/num_paths

    def _sabr_model_calibrated(self) -> Dict[str, Any]:
        if isinstance(self._udl_spot, str):
            sabr_model = SABR()
            forward = get_forward(spot=self.spot, dividend_yield=self.dividend_yield,
                                  risk_free_rate=self.risk_free_rate, maturity=self.maturity)

            sabr_calibrated_skew_params = sabr_model.calibrate_skew(ticker=self._udl_spot,
                                                                    forward=forward,
                                                                    risk_free_rate=self.risk_free_rate,
                                                                    maturity=self.maturity,
                                                                    dividend_yield=self.dividend_yield)

            sabr_model_calibrated = SABR(*sabr_calibrated_skew_params['calibrated_params'])

            if self.maturity != sabr_calibrated_skew_params['maturity_calibration']:
                warnings.warn(f"Calibration of the vol was done on options with closest maturity "
                              f"to the option one available: {sabr_calibrated_skew_params['maturity_calibration']} years "
                              f"Term structure maturity error in days is"
                              f" {abs(self.maturity - sabr_calibrated_skew_params['maturity_calibration']) * 365}")

            sabr_volatility = sabr_model_calibrated.sabr_vol(forward=forward,
                                                             strike=self.strike,
                                                             maturity=self.maturity)
            if self.option_style == 'EUR':
                sabr_model_price = self._black_scholes_price(sabr_volatility)
            else:
                sabr_model_price = self._longstaff_mc_price(volatility=sabr_volatility)

        else:
            raise ValueError('Need an underlying ticker instead of fixed spot for valid calibration '
                             'or choose and other pricing model: BS, MC or LS-MC')

        calibration_results = {'Price': sabr_model_price, 'SABR pricing volatility': sabr_volatility,
                               'Calibration params': sabr_calibrated_skew_params}
        return calibration_results

    def price(self, pricing_model: str) -> float:
        pricing_methods = {
            'Black-Scholes': {
                'EUR': lambda: self._black_scholes_price(self.volatility),
                'US': lambda: ValueError('Invalid pricing model for US style options')
            },
            'MC': {
                'EUR': lambda: self._mc_price(),
                'US': lambda: ValueError('Invalid pricing model for US style options')
            },
            'LS-MC': {
                'US': lambda: self._longstaff_mc_price(volatility=self.volatility),
                'EUR': lambda: ValueError('Invalid pricing model for EUR style options')
            }
        }

        if pricing_model in pricing_methods:
            if self.option_style in pricing_methods[pricing_model]:
                result = pricing_methods[pricing_model][self.option_style]()
                if isinstance(result, Exception):
                    raise result
                return result

        # Fallback to SABR pricing model
        return self._sabr_model_calibrated()['Price']













