import numpy as np

class RetirementAccounts:
    def __init__(self, region, tax_year=2023):
        self.region = region
        self.tax_year = tax_year
        self._initialize_account_parameters()

    def _initialize_account_parameters(self):
        if self.region == "UK":
            self._initialize_uk_parameters()
        elif self.region == "California":
            self._initialize_us_parameters()
        else:
            raise ValueError(f"Unsupported region: {self.region}")

    def _initialize_uk_parameters(self):
        self.pension_annual_allowance = 40000
        self.lifetime_allowance = 1073100

    def _initialize_us_parameters(self):
        self.traditional_401k_limit = 22500
        self.roth_401k_limit = 22500
        self.traditional_ira_limit = 6500
        self.roth_ira_limit = 6500
        self.catchup_contribution_age = 50
        self.catchup_401k = 7500
        self.catchup_ira = 1000
        self.rmd_age = 72

    def calculate_contribution(self, income, age, contribution_rate):
        if self.region == "UK":
            return self._calculate_uk_pension_contribution(income, contribution_rate)
        elif self.region == "California":
            return self._calculate_us_401k_contribution(income, age, contribution_rate)

    def _calculate_uk_pension_contribution(self, income, contribution_rate):
        contribution = income * contribution_rate
        return np.minimum(contribution, self.pension_annual_allowance)

    def _calculate_us_401k_contribution(self, income, age, contribution_rate):
        base_contribution = income * contribution_rate
        limit = self.traditional_401k_limit
        if age >= self.catchup_contribution_age:
            limit += self.catchup_401k
        return np.minimum(base_contribution, limit)

    def calculate_rmd(self, account_balance, age):
        if self.region == "UK":
            return np.zeros_like(account_balance)  # UK pensions don't have RMDs
        elif self.region == "California":
            return self._calculate_us_rmd(account_balance, age)

    def _calculate_us_rmd(self, account_balance, age):
        return np.where(age >= self.rmd_age, account_balance / (90 - age), 0)

