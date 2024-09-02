import numpy as np
import squigglepy as sq
from utils.helpers import sample_or_broadcast
from models.retirement_accounts import RetirementAccounts
from models.tax_system import TaxSystem
from models.income_paths import ARIncomePath, ConstantRealIncomePath, LinearGrowthIncomePath, ExponentialGrowthIncomePath

class PersonalFinanceModel:
    def __init__(self, input_params):
        # Initialize parameters
        self.m = input_params["m"]
        self.years = input_params["years"]
        self.r = input_params["r"]
        self.years_until_retirement = input_params["years_until_retirement"]
        self.years_until_death = input_params["years_until_death"]
        self.claim_age = input_params.get("claim_age", 67)
        self.current_age = input_params.get("current_age", 30)
        self.retirement_income = sample_or_broadcast(input_params["retirement_income"], self.m)
        
        self.income_path = input_params["income_path"]
        self.min_income = input_params["min_income"]
        self.inflation_rate = sample_or_broadcast(input_params["inflation_rate"], self.m)
        self.ar_inflation_coefficients = input_params["ar_inflation_coefficients"]
        self.ar_inflation_sd = input_params["ar_inflation_sd"]

        self.income_fraction_consumed_before_retirement = input_params.get("income_fraction_consumed_before_retirement", 0.7)
        self.income_fraction_consumed_after_retirement = input_params.get("income_fraction_consumed_after_retirement", 0.9)
        self.wealth_fraction_consumed_before_retirement = input_params.get("wealth_fraction_consumed_before_retirement", 0.05)
        self.wealth_fraction_consumed_after_retirement = sample_or_broadcast(input_params.get("wealth_fraction_consumed_after_retirement", sq.to(0.04, 0.06)), self.m)

        self.min_cash_threshold = input_params["min_cash_threshold"]
        self.max_cash_threshold = input_params["max_cash_threshold"]
        
        self.cash_start = input_params["cash_start"]
        self.market_start = input_params["market_start"]
        self.retirement_account_start = input_params.get("retirement_account_start", 0)
        
        self.tax_region = input_params["tax_region"]
        self.portfolio_weights = input_params["portfolio_weights"]
        self.asset_returns = input_params["asset_returns"]
        self.asset_volatilities = input_params["asset_volatilities"]
        self.asset_correlations = input_params["asset_correlations"]

        self.tax_system = TaxSystem(input_params["tax_region"])
        self.retirement_accounts = RetirementAccounts(input_params["tax_region"])
        
        self.retirement_contribution_rate = input_params.get("retirement_contribution_rate", 0.1)
        
        self.minimum_consumption = input_params.get("minimum_consumption", 1000)
        self.maximum_consumption_fraction = input_params.get("maximum_consumption_fraction", 2)
        
        self.initialize_arrays()

        self.charitable_giving_rate = input_params.get("charitable_giving_rate", 0.0)
        self.charitable_giving_cap = input_params.get("charitable_giving_cap", float('inf'))
        
        self.real_after_tax_income = np.zeros((self.m, self.years))
        self.real_pre_tax_income = np.zeros((self.m, self.years))
        self.real_taxable_income = np.zeros((self.m, self.years))

    def initialize_arrays(self):
        self.income = np.zeros((self.m, self.years))
        self.inflation = np.zeros((self.m, self.years))
        self.cash = np.zeros((self.m, self.years))
        self.market = np.zeros((self.m, self.years))
        self.retirement_account = np.zeros((self.m, self.years))
        self.financial_wealth = np.zeros((self.m, self.years))
        self.consumption = np.zeros((self.m, self.years))
        self.savings = np.zeros((self.m, self.years))
        self.non_financial_wealth = np.zeros((self.m, self.years))
        self.total_wealth = np.zeros((self.m, self.years))
        self.tax_paid = np.zeros((self.m, self.years))
        self.capital_gains = np.zeros((self.m, self.years))
        self.retirement_contributions = np.zeros((self.m, self.years))
        self.retirement_withdrawals = np.zeros((self.m, self.years))
        self.pension_income = np.zeros((self.m, self.years))
        self.charitable_donations = np.zeros((self.m, self.years))
        self.real_after_tax_income = np.zeros((self.m, self.years))
        self.real_pre_tax_income = np.zeros((self.m, self.years))
        self.real_taxable_income = np.zeros((self.m, self.years))

    def generate_market_returns(self):
        num_assets = len(self.portfolio_weights)
        cov_matrix = np.outer(self.asset_volatilities, self.asset_volatilities) * self.asset_correlations
        nominal_returns = np.random.multivariate_normal(self.asset_returns, cov_matrix, (self.m, self.years))
        return np.sum(nominal_returns * self.portfolio_weights, axis=2)

    def generate_ar_inflation(self):
        return ARIncomePath(self.inflation_rate, self.ar_inflation_coefficients, self.ar_inflation_sd).generate(self.years, self.m)

    def generate_income(self):
        income = self.income_path.generate(self.years, self.m)
        # Set income to zero after retirement
        retirement_mask = np.arange(self.years) >= self.years_until_retirement
        income[:, retirement_mask] = 0
        return income

    def simulate(self):
        self.market_returns = self.generate_market_returns()
        self.inflation = self.generate_ar_inflation()
        self.income = self.generate_income()
        
        cumulative_inflation = np.cumprod(1 + self.inflation, axis=1)
        real_market_returns = (1 + self.market_returns) / (1 + self.inflation) - 1

        self.income = np.maximum(self.income / cumulative_inflation, self.min_income)

        self.initialize_simulation()
        
        for t in range(self.years):
            self.simulate_year(t, real_market_returns)

        self.calculate_non_financial_wealth()
        self.check_constraints()

    def initialize_simulation(self):
        self.cash[:, 0] = self.cash_start
        self.market[:, 0] = self.market_start
        self.retirement_account[:, 0] = self.retirement_account_start
        self.financial_wealth[:, 0] = self.cash_start + self.market_start + self.retirement_account_start
        self.total_wealth[:, 0] = self.calculate_total_wealth(0, self.current_age)
        
        # Initialize consumption for the first period
        initial_income = self.income[:, 0]
        initial_wealth = self.total_wealth[:, 0]
        is_retired = np.full(self.m, False)
        years_left = self.years_until_retirement - self.current_age
        self.consumption[:, 0] = self.calculate_consumption_amount(0, initial_income, is_retired, years_left)

    def simulate_year(self, t, real_market_returns):
        current_age = self.current_age + t
        is_retired = current_age >= self.years_until_retirement + self.current_age
        years_left = self.years_until_death - (current_age - self.current_age)

        # Calculate total wealth including future pension benefits
        self.calculate_total_wealth(t, current_age)

        # Calculate income and pension
        total_real_income = self.calculate_total_income(t, current_age)
        self.real_pre_tax_income[:, t] = total_real_income
        
        # Handle retirement contributions or withdrawals
        if not is_retired:
            self.calculate_retirement_contribution(t, total_real_income, current_age)
        else:
            self.calculate_retirement_withdrawal(t, current_age, total_real_income, self.consumption[:, t-1])

        # Calculate charitable donations
        self.charitable_donations[:, t] = self.calculate_charitable_donations(t, total_real_income)

        # Calculate real after-tax income
        real_after_tax_income = self.calculate_after_tax_income(t, total_real_income)
        self.real_after_tax_income[:, t] = real_after_tax_income
        
        # Calculate consumption based on real after-tax income and total wealth
        self.consumption[:, t] = self.calculate_consumption_amount(t, real_after_tax_income, is_retired, years_left)

        # Calculate savings
        self.savings[:, t] = real_after_tax_income - self.consumption[:, t]
        
        # Update wealth
        self.update_wealth(t, real_after_tax_income, real_market_returns, is_retired)

    def calculate_total_wealth(self, t, current_age):
        financial_wealth = self.cash[:, t] + self.market[:, t]
        retirement_wealth = self.retirement_account[:, t]
        future_income = self.calculate_future_income(t)
        future_pension = self.calculate_future_pension(t, current_age)
        
        # Estimate tax on retirement account
        retirement_tax = self.estimate_retirement_account_tax(retirement_wealth, current_age)
        
        # Estimate tax on future income and pension
        future_income_tax = self.estimate_future_income_tax(future_income, future_pension, current_age)
        
        # Estimate capital gains tax
        capital_gains_tax = self.estimate_capital_gains_tax(self.market[:, t])

        self.total_wealth[:, t] = (
            financial_wealth + 
            retirement_wealth + 
            future_income + 
            future_pension -
            retirement_tax -
            future_income_tax -
            capital_gains_tax
        )
            
    def estimate_retirement_account_tax(self, retirement_wealth, current_age):
        if self.tax_region == "UK":
            # In UK, 25% of pension is tax-free
            taxable_amount = retirement_wealth * 0.75
        else:  # US
            taxable_amount = retirement_wealth
        
        # Estimate annual withdrawal over remaining lifespan
        years_left = max(1, self.years_until_death - (current_age - self.current_age))
        annual_withdrawal = taxable_amount / years_left
        
        # Calculate tax on annual withdrawal using current tax system
        annual_tax = self.tax_system.calculate_tax(annual_withdrawal, 0)
        
        return annual_tax * years_left
    
    def estimate_future_contributions(self, t, current_age):
        years_until_retirement = max(0, self.years_until_retirement - (current_age - self.current_age))
        future_income = np.sum(self.income[:, t:t+years_until_retirement], axis=1)
        return future_income * self.retirement_contribution_rate
    
    def estimate_future_income_tax(self, future_income, future_pension, current_age):
        years_left = max(1, self.years_until_death - (current_age - self.current_age))
        annual_income = (future_income + future_pension) / years_left
        
        # Estimate future contributions
        years_until_retirement = max(0, self.years_until_retirement - (current_age - self.current_age))
        annual_contribution = (future_income / years_left) * self.retirement_contribution_rate * (years_until_retirement / years_left)
        
        # Calculate tax on annual income using current tax system, accounting for contributions
        annual_tax = self.tax_system.calculate_tax((annual_income - annual_contribution), 0)
        
        return annual_tax * years_left

    def estimate_capital_gains_tax(self, market_value):
        # Estimate capital gains as a percentage of market value
        estimated_gains = market_value * 0.5  # Assume 50% of market value is gains
        
        # Calculate capital gains tax using current tax system
        if self.tax_region == "UK":
            # UK has a separate capital gains tax calculation
            cgt = self.tax_system.calculate_tax(0, estimated_gains)
        else:  # US
            # In the US, capital gains are part of the regular tax calculation
            cgt = self.tax_system.calculate_tax(estimated_gains, estimated_gains)
            cgt -= self.tax_system.calculate_tax(estimated_gains, 0)
        
        return cgt

    def calculate_total_income(self, t, current_age):
        base_income = self.income[:, t]
        self.pension_income[:, t] = self.calculate_pension_income(t, current_age)
        
        total_income = base_income + self.pension_income[:, t]
        
        return np.maximum(total_income, self.min_income)

    def calculate_retirement_withdrawal(self, t, current_age, total_real_income, consumption):
        required_withdrawal = np.maximum(consumption - total_real_income, 0)
        
        # Calculate proportions of pension and non-pension wealth
        pension_wealth = self.calculate_future_pension(t, current_age)
        non_pension_wealth = np.maximum(self.total_wealth[:, t] - pension_wealth, 0)  # Ensure non-negative
        total_wealth_safe = np.maximum(self.total_wealth[:, t], 1e-10)  # Avoid division by zero
        pension_proportion = np.clip(pension_wealth / total_wealth_safe, 0, 1)
        non_pension_proportion = np.clip(non_pension_wealth / total_wealth_safe, 0, 1)

        # Calculate withdrawals from each source
        pension_withdrawal = required_withdrawal * pension_proportion
        non_pension_withdrawal = required_withdrawal * non_pension_proportion

        # Ensure we don't withdraw more than available from retirement account
        max_retirement_withdrawal = self.retirement_account[:, t]
        actual_retirement_withdrawal = np.minimum(non_pension_withdrawal, max_retirement_withdrawal)

        total_withdrawal = pension_withdrawal + actual_retirement_withdrawal

        self.retirement_withdrawals[:, t] = actual_retirement_withdrawal

    def calculate_pension_income(self, t, current_age):
        if self.tax_region == "UK":
            return self.calculate_uk_pension(t, current_age)
        elif self.tax_region in ["California", "Massachusetts", "New York", "DC", "Texas"]:
            return self.calculate_us_social_security(t, current_age)
        else:
            return np.zeros(self.m)

    def calculate_uk_pension(self, t, current_age):
        pension_amount = np.where(
            current_age >= self.claim_age,
            (np.minimum(self.claim_age - self.current_age, self.tax_system.uk_qualifying_years) / self.tax_system.uk_qualifying_years) * self.tax_system.uk_full_pension,
            0
        )
        return pension_amount

    def calculate_us_social_security(self, t, current_age):
        if np.all(current_age < self.claim_age):
            return np.zeros(self.m)

        # Calculate AIME (Average Indexed Monthly Earnings)
        max_taxable_earnings = self.tax_system.max_taxable_earnings
        indexed_earnings = np.minimum(self.income[:, :self.years_until_retirement], max_taxable_earnings)
        aime = np.mean(indexed_earnings, axis=1) / 12

        # Calculate PIA (Primary Insurance Amount)
        pia = np.zeros(self.m)
        for i, (bend_point, factor) in enumerate(zip(self.tax_system.bend_points, self.tax_system.pia_factors)):
            if i == 0:
                pia += np.minimum(aime, bend_point) * factor
            elif i == len(self.tax_system.bend_points) - 1:
                pia += np.maximum(0, aime - bend_point) * factor
            else:
                pia += np.maximum(0, np.minimum(aime - self.tax_system.bend_points[i-1], bend_point - self.tax_system.bend_points[i-1])) * factor

        # Adjust for claiming age
        months_diff = (self.claim_age - self.tax_system.fra) * 12
        if self.claim_age < self.tax_system.fra:
            age_adjustment = 1 - 0.00555556 * np.minimum(36, months_diff) - 0.00416667 * np.maximum(0, months_diff - 36)
        else:
            age_adjustment = 1 + 0.00666667 * months_diff

        pia *= age_adjustment

        # Apply maximum benefit limit (example values for 2023, should be updated yearly)
        max_benefit = np.where(self.claim_age == 62, 2572,
                            np.where(self.claim_age == self.tax_system.fra, 3627,
                                        np.where(self.claim_age == 70, 4555, 3627)))

        pia = np.minimum(pia, max_benefit)

        # Return the annual benefit in real terms
        return np.where(current_age >= self.claim_age, pia * 12, 0)
    
    def calculate_consumption_amount(self, t, total_real_income, is_retired, years_left):
        wealth_consumption_rate = np.where(is_retired, self.wealth_fraction_consumed_after_retirement, self.wealth_fraction_consumed_before_retirement)
        income_consumption_rate = np.where(is_retired, self.income_fraction_consumed_after_retirement, self.income_fraction_consumed_before_retirement)
        
        # Calculate annualized wealth
        annualized_wealth = np.where(years_left > 0, self.total_wealth[:, t] / years_left, self.total_wealth[:, t])
        
        # Calculate consumption from annualized wealth and income
        consumption_from_wealth = wealth_consumption_rate * annualized_wealth
        consumption_from_income = income_consumption_rate * total_real_income
        
        base_consumption = consumption_from_income + consumption_from_wealth
        
        min_consumption = self.minimum_consumption / (1 + self.inflation[:, t])**t
        
        # Ensure max_consumption doesn't exceed available resources
        available_resources = total_real_income + self.total_wealth[:, t]
        max_consumption = np.minimum(self.maximum_consumption_fraction * available_resources, available_resources)
        
        return np.clip(base_consumption, min_consumption, max_consumption)

    def update_wealth(self, t, after_tax_income, real_market_returns, is_retired):
        # Calculate total savings
        total_savings = self.savings[:, t]

        # Update retirement account
        self.retirement_account[:, t] *= (1 + real_market_returns[:, t])
        if not is_retired:
            self.retirement_account[:, t] += self.retirement_contributions[:, t]
        else:
            self.retirement_account[:, t] -= self.retirement_withdrawals[:, t]

        # Calculate capital gains
        self.capital_gains[:, t] = self.market[:, t] * real_market_returns[:, t]

        # Update market value
        self.market[:, t] *= (1 + real_market_returns[:, t])

        # Add savings to cash
        self.cash[:, t] += total_savings

        # Adjust cash and market
        self.adjust_cash_and_market(t)

        # Update financial wealth
        self.financial_wealth[:, t] = self.cash[:, t] + self.market[:, t] + self.retirement_account[:, t]
        
        if t < self.years - 1:
            self.cash[:, t+1] = self.cash[:, t]
            self.market[:, t+1] = self.market[:, t]
            self.retirement_account[:, t+1] = self.retirement_account[:, t]
            self.financial_wealth[:, t+1] = self.financial_wealth[:, t]

    def calculate_retirement_contribution(self, t, total_real_income, current_age):
        contribution = np.minimum(
            self.retirement_accounts.calculate_contribution(
                total_real_income,
                current_age,
                self.retirement_contribution_rate
            ),
            total_real_income * 0.9
        )
        self.retirement_contributions[:, t] = contribution
        return contribution

    def calculate_after_tax_income(self, t, total_real_income):
        real_capital_gains = self.capital_gains[:, t]
        real_donations = self.charitable_donations[:, t]
        real_contributions = self.retirement_contributions[:, t]
        real_withdrawals = self.retirement_withdrawals[:, t]

        # Calculate taxable income in real terms
        real_taxable_income = total_real_income - real_contributions - real_donations + real_withdrawals
        self.real_taxable_income[:, t] = real_taxable_income

        # Calculate tax using real numbers
        self.tax_paid[:, t] = self.tax_system.calculate_tax(real_taxable_income, real_capital_gains, real_donations)

        # Calculate and return real after-tax income
        real_after_tax_income = total_real_income - self.tax_paid[:, t]
        return real_after_tax_income

    def adjust_cash_and_market(self, t):
        total_liquid = self.cash[:, t] + self.market[:, t]
        
        # If cash is below minimum, attempt to transfer from market to cash
        cash_deficit = self.min_cash_threshold - self.cash[:, t]
        cash_deficit = np.where(cash_deficit > 0, cash_deficit, 0)
        transfer_to_cash = np.minimum(cash_deficit, self.market[:, t])
        self.cash[:, t] += transfer_to_cash
        self.market[:, t] -= transfer_to_cash

        # If cash exceeds maximum, move excess to market
        excess_cash = self.cash[:, t] - self.max_cash_threshold
        excess_cash = np.where(excess_cash > 0, excess_cash, 0)
        self.cash[:, t] -= excess_cash
        self.market[:, t] += excess_cash

    def calculate_future_income(self, t):
        return np.sum(self.income[:, t:], axis=1)

    def calculate_future_pension(self, t, current_age):
        years_until_claim = max(0, self.claim_age - current_age)
        future_pension = np.sum(self.pension_income[:, t+years_until_claim:], axis=1)
        return future_pension

    def calculate_non_financial_wealth(self):
        for t in range(self.years):
            current_age = self.current_age + t
            self.non_financial_wealth[:, t] = (self.calculate_future_income(t) + 
                                               self.calculate_future_pension(t, current_age))

    def calculate_charitable_donations(self, t, total_real_income):
        donations = total_real_income * self.charitable_giving_rate
        return np.minimum(donations, self.charitable_giving_cap)

    def calculate_after_tax_income(self, t, total_real_income):
        capital_gains = self.capital_gains[:, t]
        donations = self.charitable_donations[:, t]

        # Calculate retirement contributions
        contribution = self.calculate_retirement_contribution(t, total_real_income, self.current_age + t)

        # Subtract retirement contributions and charitable donations from taxable income
        taxable_income = total_real_income - contribution - donations

        self.tax_paid[:, t] = self.tax_system.calculate_tax(taxable_income, capital_gains, donations)
        return total_real_income - self.tax_paid[:, t] - contribution

    def get_results(self):
        return {
            "income": self.income,
            "pension_income": self.pension_income,
            "inflation": self.inflation,
            "cash": self.cash,
            "market": self.market,
            "retirement_account": self.retirement_account,
            "financial_wealth": self.financial_wealth,
            "consumption": self.consumption,
            "savings": self.savings,
            "non_financial_wealth": self.non_financial_wealth,
            "total_wealth": self.total_wealth,
            "tax_paid": self.tax_paid,
            "capital_gains": self.capital_gains,
            "retirement_contributions": self.retirement_contributions,
            "retirement_withdrawals": self.retirement_withdrawals,
            "charitable_donations": self.charitable_donations,
            "real_after_tax_income": self.real_after_tax_income,
            "real_pre_tax_income": self.real_pre_tax_income
        }

    def check_constraints(self):
        violations = []
        violations.extend(self.check_cash_constraints())
        violations.extend(self.check_market_constraints())
        violations.extend(self.check_retirement_account_constraints())
        violations.extend(self.check_financial_wealth_constraints())
        violations.extend(self.check_consumption_constraints())
        violations.extend(self.check_savings_constraints())
        violations.extend(self.check_liquid_asset_conservation())

        if violations:
            raise ValueError("Simulation failed due to constraint violations:\n" + "\n".join(violations))

    def check_cash_constraints(self):
        violations = []
        for t in range(self.years):
            cash_below_min = np.sum(self.cash[:, t] < self.min_cash_threshold)
            if cash_below_min > 0:
                violations.append(f"At time {t}, {cash_below_min} cash values below minimum threshold")
        return violations

    def check_market_constraints(self):
        violations = []
        for t in range(self.years):
            negative_market = np.sum(self.market[:, t] < 0)
            if negative_market > 0:
                violations.append(f"At time {t}, {negative_market} negative market values detected")
        return violations

    def check_retirement_account_constraints(self):
        violations = []
        for t in range(self.years):
            negative_retirement = np.sum(self.retirement_account[:, t] < 0)
            if negative_retirement > 0:
                violations.append(f"At time {t}, {negative_retirement} negative retirement account values detected")
        return violations

    def check_financial_wealth_constraints(self):
        violations = []
        for t in range(self.years):
            negative_wealth = np.sum(self.financial_wealth[:, t] < 0)
            if negative_wealth > 0:
                violations.append(f"At time {t}, {negative_wealth} negative financial wealth values detected")
        return violations

    def check_consumption_constraints(self):
        violations = []
        for t in range(self.years):
            negative_consumption = np.sum(self.consumption[:, t] < 0)
            if negative_consumption > 0:
                violations.append(f"At time {t}, {negative_consumption} negative consumption values detected")
        return violations

    def check_savings_constraints(self):
        violations = []
        for t in range(self.years):
            negative_savings = np.sum(self.savings[:, t] < 0)
            if negative_savings > 0:
                violations.append(f"At time {t}, {negative_savings} negative savings values detected")
        return violations

    def check_liquid_asset_conservation(self):
        violations = []
        for t in range(1, self.years):
            prev_liquid = self.cash[:, t-1] + self.market[:, t-1]
            current_liquid = self.cash[:, t] + self.market[:, t]
            savings = self.savings[:, t-1]
            market_returns = self.market[:, t-1] * self.market_returns[:, t-1]
            expected_liquid = prev_liquid + savings + market_returns
            if not np.allclose(current_liquid, expected_liquid, rtol=1e-5, atol=1e-8):
                violations.append(f"At time {t}, liquid assets not conserved")
        return violations
