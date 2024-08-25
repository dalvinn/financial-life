import numpy as np

class TaxSystem:
    def __init__(self, region, tax_year=2023):
        self.region = region
        self.tax_year = tax_year
        self._initialize_tax_parameters()

    def _initialize_tax_parameters(self):
        if self.region == "UK":
            self._initialize_uk_parameters()
        elif self.region == "California":
            self._initialize_california_parameters()
        else:
            raise ValueError(f"Unsupported tax region: {self.region}")

    def _initialize_uk_parameters(self):
        self.personal_allowance = 12570.0
        self.basic_rate_threshold = 50270.0
        self.higher_rate_threshold = 150000.0
        self.additional_rate_threshold = 125140.0
        self.basic_rate = 0.20
        self.higher_rate = 0.40
        self.additional_rate = 0.45
        self.ni_primary_threshold = 9568.0
        self.ni_upper_earnings_limit = 50270.0
        self.ni_basic_rate = 0.12
        self.ni_higher_rate = 0.02
        self.capital_gains_allowance = 6000.0
        self.capital_gains_basic_rate = 0.10
        self.capital_gains_higher_rate = 0.20
        self.uk_full_pension = 10600.0
        self.uk_qualifying_years = 35.0

    def _initialize_california_parameters(self):
        self.federal_brackets = [
            (0.0, 0.10), (9950.0, 0.12), (40525.0, 0.22),
            (86375.0, 0.24), (164925.0, 0.32), (209425.0, 0.35),
            (523600.0, 0.37)
        ]
        self.ca_brackets = [
            (0.0, 0.01), (8932.0, 0.02), (21175.0, 0.04),
            (33421.0, 0.06), (46394.0, 0.08), (58634.0, 0.093),
            (299508.0, 0.103), (359407.0, 0.113), (599012.0, 0.123)
        ]
        self.ss_wage_base = 142800.0
        self.ss_rate = 0.062
        self.medicare_rate = 0.0145
        self.capital_gains_brackets = [
            (0.0, 0.0), (40400.0, 0.15), (445850.0, 0.20)
        ]
        self.max_taxable_earnings = 160200.0
        self.bend_points = [1115.0, 6721.0]
        self.pia_factors = [0.9, 0.32, 0.15]
        self.fra = 67.0

    def calculate_tax(self, income, capital_gains=0):
        if self.region == "UK":
            return self._calculate_uk_tax(income, capital_gains)
        elif self.region == "California":
            return self._calculate_california_tax(income, capital_gains)

    def _calculate_uk_tax(self, income, capital_gains):
        taxable_income = np.maximum(income - self.personal_allowance, 0.0)
        basic_rate_tax = np.minimum(taxable_income, self.basic_rate_threshold - self.personal_allowance) * self.basic_rate
        higher_rate_tax = np.maximum(np.minimum(taxable_income, self.higher_rate_threshold) - (self.basic_rate_threshold - self.personal_allowance), 0.0) * self.higher_rate
        additional_rate_tax = np.maximum(taxable_income - self.higher_rate_threshold, 0.0) * self.additional_rate

        total_income_tax = basic_rate_tax + higher_rate_tax + additional_rate_tax
        
        ni_contributions = np.minimum(np.maximum(income - self.ni_primary_threshold, 0.0), self.ni_upper_earnings_limit - self.ni_primary_threshold) * self.ni_basic_rate + np.maximum(income - self.ni_upper_earnings_limit, 0.0) * self.ni_higher_rate

        taxable_capital_gains = np.maximum(capital_gains - self.capital_gains_allowance, 0.0)
        capital_gains_tax = np.minimum(taxable_capital_gains, np.maximum(self.basic_rate_threshold - income, 0)) * self.capital_gains_basic_rate + np.maximum(taxable_capital_gains - np.maximum(self.basic_rate_threshold - income, 0), 0.0) * self.capital_gains_higher_rate

        return total_income_tax + ni_contributions + capital_gains_tax
    
    def _calculate_california_tax(self, income, capital_gains):
        federal_income_tax = self._calculate_bracketed_tax(income, self.federal_brackets)
        ca_income_tax = self._calculate_bracketed_tax(income, self.ca_brackets)
        ss_tax = np.minimum(income, self.ss_wage_base) * self.ss_rate
        medicare_tax = income * self.medicare_rate
        capital_gains_tax = self._calculate_bracketed_tax(capital_gains, self.capital_gains_brackets)

        return federal_income_tax + ca_income_tax + ss_tax + medicare_tax + capital_gains_tax

    def _calculate_bracketed_tax(self, amount, brackets):
        tax = np.zeros_like(amount, dtype=np.float64)
        for i, (threshold, rate) in enumerate(brackets):
            if i == len(brackets) - 1:
                tax += np.maximum(amount - threshold, 0.0) * rate
            else:
                next_threshold = brackets[i+1][0]
                tax += np.minimum(np.maximum(amount - threshold, 0.0), next_threshold - threshold) * rate
        return tax

