from squigglepy import K, M

input_params = {
    "m": 200,  # The number of paths
    "years": 70,
    "cash_start": 10 * K,
    "market_start": 10 * K,
    "life_cycle_income": [
        (40 * K) + (year * 2 * K) for year in range(70)
    ],  # Baseline income lifecycle
    "min_income": 15 * K,
    "min_cash": 0,
    "min_market": 0,

    "income_fraction_consumed": 0.4,
    "wealth_fraction_consumed_before_retirement": 0.35,
    "wealth_fraction_consumed_after_retirement": 0.45,

    "max_cash_threshold": 10 * K,
    "min_cash_threshold": 5 * K,

    "inflation_rate": 0.02,
    "ar_income_coefficients": [0.4, 0.2, 0.1],
    "ar_income_sd": 15 * K,
    "ar_inflation_coefficients": [0.4, 0.2],
    "ar_inflation_sd": 0.01,
    "r": 0.02,  # Discount rate

    "years_until_retirement": 45,
    "years_until_death": 70,
    "retirement_income": 10 * K,
}
