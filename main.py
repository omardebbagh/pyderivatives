from pyderivatives.products.vanilla_options import VanillaOption

PRICING_MODELS = ['Black-Scholes', 'MC', 'LS-MC', 'SABR Vol']


def main():

    ticker_or_spot = "AAPL"  # APPLE ticker for example - can put fixed spot instead
    maturity = 1.5
    strike = 200
    risk_free_rate = 0.04
    option_type = 'put'  # put
    option_style = 'EUR'  # US
    sigma = None  # need to have value if fixed spot instead of ticker
    div_yield = None

    option = VanillaOption(udl_spot=ticker_or_spot, strike=strike, maturity=maturity,
                           risk_free_rate=risk_free_rate, sigma=sigma, div_yield=div_yield,
                           option_type=option_type, option_style=option_style)

    # valid pricing models for only EUR style options
    option.option_style = 'EUR'
    option_bs_price = option.price(pricing_model=PRICING_MODELS[0])
    print(f"BS option price: {option_bs_price}")
    option_mc_price = option.price(pricing_model=PRICING_MODELS[1])
    print(f"MC option price: {option_mc_price}")
    option_sabr_price = option.price(pricing_model=PRICING_MODELS[3])
    print(f"SABR option price EUR style: {option_sabr_price}")  # valid pricing model for both US and EUR options

    # valid pricing model for only US style options
    option.option_style = 'US'
    option_ls_mc_price = option.price(pricing_model=PRICING_MODELS[2])
    print(f"LS-MC option price: {option_ls_mc_price}")
    option_sabr_price = option.price(pricing_model=PRICING_MODELS[3])
    print(f"SABR option price US style: {option_sabr_price}")  # valid pricing model for both US and EUR options


if __name__ == "__main__":
    main()
