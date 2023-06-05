from squigglepy import K, M

input_params = {
    "m": 300,  # The number of paths
    "years": 25,
    "cash_start": 35 * K,
    "market_start": 150 * K,
    "life_cycle_income": [
        (65 * K) + (year * 3 * K) for year in range(25)
    ],  # Baseline income lifecycle
    "min_income": 30 * K,
    "min_cash": 0,
    "min_market": 0,
    "inflation_rate": 0.02,
    "ar_income_coefficients": [0.5, 0.3, 0.1],
    "ar_income_sd": 15 * K,
    "ar_inflation_coefficients": [0.4, 0.2],
    "ar_inflation_sd": 0.01,
    "r": 0.02,  # Discount rate
    "proportion_dissavings_from_cash": 0.06,
    "beta": 0.992,  # Rate of time preference
    "alpha": 1.5,  # CRRA parameter
}
