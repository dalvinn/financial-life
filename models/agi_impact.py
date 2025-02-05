import numpy as np
from scipy import interpolate


class AGIModel:
    def __init__(self, cdf_points, wage_multiplier, returns_multiplier, m, years):
        """
        Initialize AGI model with CDF points and impact parameters.

        Args:
            cdf_points: dict with years as keys and cumulative probabilities as values
            wage_multiplier: Factor by which wages change after AGI
            returns_multiplier: Factor by which investment returns change after AGI
            m: Number of simulations
            years: Number of years to simulate
        """
        self.cdf_points = cdf_points
        self.wage_multiplier = wage_multiplier
        self.returns_multiplier = returns_multiplier
        self.m = m
        self.years = years

        # Create interpolated CDF and PDF
        self._setup_distributions()

        # Generate AGI timing for each simulation
        self.agi_timing = self._generate_agi_timing()

        # Create multiplier arrays
        self.wage_multipliers = self._create_multiplier_array(self.wage_multiplier)
        self.returns_multipliers = self._create_multiplier_array(
            self.returns_multiplier
        )

    def _setup_distributions(self):
        """Create interpolated CDF and PDF from provided points."""
        years = np.array(list(self.cdf_points.keys()))
        probs = np.array(list(self.cdf_points.values()))

        # Add endpoints if not provided
        if 0 not in years:
            years = np.insert(years, 0, 0)
            probs = np.insert(probs, 0, 0)
        if 100 not in years:  # Far future endpoint
            years = np.append(years, 100)
            probs = np.append(probs, 1)

        # Create interpolated CDF
        self.cdf = interpolate.interp1d(years, probs, kind="linear")

        # Approximate PDF using finite differences
        dx = 0.1
        x_pdf = np.arange(0, 100, dx)
        y_cdf = self.cdf(x_pdf)
        self.pdf = np.gradient(y_cdf, dx)
        self.pdf_x = x_pdf

    def _generate_agi_timing(self):
        """Generate AGI arrival times for each simulation."""
        # Generate uniform random numbers
        u = np.random.uniform(0, 1, self.m)

        # Create inverse CDF points
        x = np.array(list(self.cdf_points.keys()))
        y = np.array(list(self.cdf_points.values()))

        # Add endpoints if not provided
        if 0 not in x:
            x = np.insert(x, 0, 0)
            y = np.insert(y, 0, 0)
        if 100 not in x:
            x = np.append(x, 100)
            y = np.append(y, 1)

        # Create inverse CDF interpolation
        inv_cdf = interpolate.interp1d(
            y, x, kind="linear", bounds_error=False, fill_value=(0, 100)
        )

        # Get AGI timing for each simulation
        agi_timing = inv_cdf(u)

        return np.floor(agi_timing).astype(int)

    def _create_multiplier_array(self, multiplier):
        """Create array of multipliers for each simulation and year."""
        multipliers = np.ones((self.m, self.years))
        for i in range(self.m):
            if self.agi_timing[i] < self.years:
                multipliers[i, self.agi_timing[i] :] = multiplier
        return multipliers

    def get_wage_multipliers(self):
        """Get array of wage multipliers for each simulation and year."""
        return self.wage_multipliers

    def get_returns_multipliers(self):
        """Get array of returns multipliers for each simulation and year."""
        return self.returns_multipliers

    def get_agi_timing(self):
        """Get array of AGI timing for each simulation."""
        return self.agi_timing
