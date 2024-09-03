import numpy as np
from abc import ABC, abstractmethod
from utils.helpers import sample_or_broadcast

class IncomePath(ABC):
    @abstractmethod
    def generate(self, years, m):
        pass

class ARIncomePath(IncomePath):
    def __init__(self, baseline, ar_coefficients, sd):
        self.baseline = baseline
        self.ar_coefficients = ar_coefficients
        self.sd = sd

    def generate(self, years, m):
        baseline = sample_or_broadcast(self.baseline, m)
        sd = sample_or_broadcast(self.sd, m)
        
        p = len(self.ar_coefficients)
        result = np.zeros((m, years))
        shocks = np.random.normal(0, sd[:, np.newaxis], (m, years))
        
        result[:, 0] = baseline + shocks[:, 0]
        for t in range(1, years):
            ar_terms = np.sum([coef * (result[:, t-j-1] - baseline) for j, coef in enumerate(self.ar_coefficients[:min(t, p)])], axis=0)
            result[:, t] = baseline + ar_terms + shocks[:, t]
        
        return result

class ConstantRealIncomePath(IncomePath):
    def __init__(self, base_income, sd):
        self.base_income = base_income
        self.sd = sd

    def generate(self, years, m):
        base_income = sample_or_broadcast(self.base_income, m)
        sd = sample_or_broadcast(self.sd, m)
        constant_income = np.tile(base_income[:, np.newaxis], (1, years))
        shocks = np.random.normal(0, sd[:, np.newaxis], (m, years))
        return constant_income + shocks

class LinearGrowthIncomePath(IncomePath):
    def __init__(self, base_income, annual_growth_rate, sd):
        self.base_income = base_income
        self.annual_growth_rate = annual_growth_rate
        self.sd = sd

    def generate(self, years, m):
        base_income = sample_or_broadcast(self.base_income, m)
        annual_growth_rate = sample_or_broadcast(self.annual_growth_rate, m)
        sd = sample_or_broadcast(self.sd, m)
        growth_factors = np.outer(annual_growth_rate, np.arange(years))
        linear_growth = base_income[:, np.newaxis] * (1 + growth_factors)
        shocks = np.random.normal(0, sd[:, np.newaxis], (m, years))
        return linear_growth + shocks

class ExponentialGrowthIncomePath(IncomePath):
    def __init__(self, base_income, annual_growth_rate, sd):
        self.base_income = base_income
        self.annual_growth_rate = annual_growth_rate
        self.sd = sd

    def generate(self, years, m):
        base_income = sample_or_broadcast(self.base_income, m)
        annual_growth_rate = sample_or_broadcast(self.annual_growth_rate, m)
        sd = sample_or_broadcast(self.sd, m)
        growth_factors = (1 + annual_growth_rate[:, np.newaxis]) ** np.arange(years)
        exponential_growth = base_income[:, np.newaxis] * growth_factors
        shocks = np.random.normal(0, sd[:, np.newaxis], (m, years))
        return exponential_growth + shocks

