import numpy as np
from models.personal_finance import PersonalFinanceModel
from models.income_paths import ExponentialGrowthIncomePath

def print_step(step_name, result):
    print(f"\n--- {step_name} ---")
    if isinstance(result, dict):
        for key, value in result.items():
            print(f"  {key}: {value}")
    elif isinstance(result, (int, float, np.number)):
        print(f"  Result: {result}")
    elif isinstance(result, np.ndarray):
        print(f"  Result shape: {result.shape}")
        print(f"  Result content: {result}")
    else:
        print(f"  Result: {result}")

def print_array_info(name, arr):
    print(f"{name} shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"{name} content: {arr}")

def run_step_by_step_simulation(start_year, num_years):
    # Example input parameters
    input_params = {
        "m": 1,  # Number of simulations
        "years": 2,
        "r": 0.02,
        "years_until_retirement": 30,
        "years_until_death": 60,
        "claim_age": 67,
        "current_age": 30,
        "retirement_income": 40000,
        "income_path": ExponentialGrowthIncomePath(100000, 0.02),
        "min_income": 30000,
        "inflation_rate": 0.02,
        "ar_inflation_coefficients": [0],
        "ar_inflation_sd": 0,
        "income_fraction_consumed_before_retirement": 1.0,
        "income_fraction_consumed_after_retirement": 1.0,
        "wealth_fraction_consumed_before_retirement": 0,
        "wealth_fraction_consumed_after_retirement": 0,
        "min_cash_threshold": 0,
        "max_cash_threshold": 0,
        "cash_start": 0,
        "market_start": 0,
        "retirement_account_start": 0,
        "retirement_contribution_rate": 0.05,
        "charitable_giving_rate": 0,
        "charitable_giving_cap": 0,
        "tax_region": "California",
        "portfolio_weights": [1.0],
        "asset_returns": [0.02],
        "asset_volatilities": [0],
        "asset_correlations": [[1.0]]
    }

    model = PersonalFinanceModel(input_params)

    print("\n=== Initializing Simulation ===")
    
    print_step("Generate Market Returns", model.generate_market_returns())
    model.market_returns = model.generate_market_returns()
    
    print_step("Generate AR Inflation", model.generate_ar_inflation())
    model.inflation = model.generate_ar_inflation()
    
    print_step("Generate Income", model.generate_income())
    model.income = model.generate_income()

    print_step("Calculate Cumulative Inflation", np.cumprod(1 + model.inflation, axis=1))
    cumulative_inflation = np.cumprod(1 + model.inflation, axis=1)
    
    print_step("Calculate Real Market Returns", (1 + model.market_returns) / (1 + model.inflation) - 1)
    real_market_returns = (1 + model.market_returns) / (1 + model.inflation) - 1

    print_step("Adjust Income for Inflation", np.maximum(model.income / cumulative_inflation, model.min_income / cumulative_inflation))
    model.income = np.maximum(model.income / cumulative_inflation, model.min_income / cumulative_inflation)

    print_step("Initialize Simulation", None)
    model.initialize_simulation()

    for year in range(start_year, start_year + num_years):
        print(f"\n\n=== Simulating Year {year} ===")
        try:
            model.simulate_year(year, real_market_returns)
            
            print_step("Year Summary", {
                "Age": model.current_age + year,
                "Income": model.income[0, year],
                "Inflation": model.inflation[0, year],
                "Market Returns": model.market_returns[0, year],
                "Cash": model.cash[0, year],
                "Market Investments": model.market[0, year],
                "Retirement Account": model.retirement_account[0, year],
                "Financial Wealth": model.financial_wealth[0, year],
                "Retirement Withdrawals": model.retirement_withdrawals[0, year],
                "Retirement Contributions": model.retirement_contributions[0, year],
                "Total Wealth": model.total_wealth[0, year],
                "Consumption": model.consumption[0, year],
                "Savings": model.savings[0, year],
                "Cash Savings": model.cash_savings[0, year],
                "Market Savings": model.market_savings[0, year],
                "Tax Paid": model.tax_paid[0, year],
                "Charitable Donations": model.charitable_donations[0, year],
                "Real After-Tax Income": model.real_after_tax_income[0, year],
                "Real Pre-Tax Income": model.real_pre_tax_income[0, year],
                "Real Taxable Income": model.real_taxable_income[0, year],
            })

        except Exception as e:
            print(f"Error in year {year}: {str(e)}")
            print_array_info("cash", model.cash)
            print_array_info("market", model.market)
            print_array_info("retirement_account", model.retirement_account)
            print_array_info("financial_wealth", model.financial_wealth)
            raise

        input("Press Enter to continue to the next year...")

    print("\n=== Calculating Non-Financial Wealth ===")
    model.calculate_non_financial_wealth()
    print_step("Non-Financial Wealth", model.non_financial_wealth)

    print("\n=== Simulation Complete ===")

if __name__ == "__main__":
    run_step_by_step_simulation(start_year=0, num_years=2)