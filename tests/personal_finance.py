import unittest
import numpy as np
from models.personal_finance import PersonalFinanceModel

class TestPersonalFinanceModel(unittest.TestCase):
    def setUp(self):
        # Set up a basic model for testing
        self.input_params = {
            "m": 1000,
            "years": 40,
            "r": 0.05,
            "years_until_retirement": 30,
            "years_until_death": 60,
            "retirement_income": 50000,
            "income_path": None,  # You'll need to provide a mock or real income path
            "min_income": 20000,
            "inflation_rate": 0.02,
            "ar_inflation_coefficients": [0.5],
            "ar_inflation_sd": 0.01,
            "min_cash_threshold": 5000,
            "max_cash_threshold": 20000,
            "cash_start": 10000,
            "market_start": 50000,
            "tax_region": "US",
            "portfolio_weights": [1.0],  # Simplified for testing
            "asset_returns": [0.07],
            "asset_volatilities": [0.15],
            "asset_correlations": [[1.0]]
        }
        self.model = PersonalFinanceModel(self.input_params)

    def test_initialization(self):
        self.assertEqual(self.model.m, 1000)
        self.assertEqual(self.model.years, 40)
        self.assertEqual(self.model.r, 0.05)

    def test_generate_market_returns(self):
        returns = self.model.generate_market_returns()
        self.assertEqual(returns.shape, (1000, 40))
        self.assertTrue(np.all(returns > -1))  # Returns should be greater than -100%

    def test_generate_ar_inflation(self):
        inflation = self.model.generate_ar_inflation()
        self.assertEqual(inflation.shape, (1000, 40))
        self.assertTrue(np.all(inflation > 0))  # Inflation should be positive

    def test_simulate(self):
        self.model.simulate()
        results = self.model.get_results()
        
        # Check that all result arrays have the correct shape
        for key, value in results.items():
            self.assertEqual(value.shape, (1000, 40), f"{key} has incorrect shape")

        # Check that financial wealth is non-negative
        self.assertTrue(np.all(results['financial_wealth'] >= 0))

        # Check that consumption is always positive
        self.assertTrue(np.all(results['consumption'] > 0))

    def test_calculate_charitable_donations(self):
        self.model.charitable_giving_rate = 0.05
        self.model.charitable_giving_cap = 10000
        total_real_income = np.array([100000, 200000, 300000])
        donations = self.model.calculate_charitable_donations(0, total_real_income)
        expected_donations = np.array([5000, 10000, 10000])  # Cap should be applied to last value
        np.testing.assert_array_almost_equal(donations, expected_donations)

if __name__ == '__main__':
    unittest.main()