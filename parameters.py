from squigglepy import K, M

input_params = {
    "m": 300,  # The number of paths
    "years": 70,
    "cash_start": 10 * K,
    "market_start": 30 * K,
    "life_cycle_income": [
        (40 * K) + (year * 3 * K) for year in range(70)
    ],  # Baseline income lifecycle
    "min_income": 10 * K,
    "min_cash": 0,
    "min_market": 0,

    "income_fraction_consumed": 0.9,
    "wealth_fraction_consumed_before_retirement": 0.9,
    "wealth_fraction_consumed_after_retirement": 1.0,

    "max_cash_threshold": 15 * K,
    "min_cash_threshold": 5 * K,

    "inflation_rate": 0.02,
    "ar_income_coefficients": [0.5, 0.3, 0.1],
    "ar_income_sd": 15 * K,
    "ar_inflation_coefficients": [0.4, 0.2],
    "ar_inflation_sd": 0.01,
    "r": 0.02,  # Discount rate
    "proportion_dissavings_from_cash": 0.06,
    "beta": 0.992,  # Rate of time preference
    "alpha": 1.5,  # CRRA parameter

    "years_until_retirement": 45,
    "years_until_death": 70,
    "retirement_income": 10 * K,
}
