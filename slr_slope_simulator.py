import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from plotnine import ggplot, aes, geom_histogram, labs

class SLR_slope_simulator:
    """
    Simulate the sampling distribution of the estimated slope in simple linear regression.  
    Uses the model: Y_i = beta_0 + beta_1 * x_i + E_i, where E_i ~ Normal(0, sigma^2).
    """
    def __init__(self, beta_0, beta_1, x, sigma, seed):
        """Initialize the simulator with model parameters, predictor values, and a random seed."""
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.sigma = sigma
        self.x = x
        self.n = len(x)
        self.rng = np.random.default_rng(seed)
        self.slopes = []

    def generate_data(self):
        """Generate one simulated dataset of (x, y) using the SLR model with normal random errors."""
        y = self.beta_0 + self.beta_1 * self.x + self.rng.normal(loc=0, scale=self.sigma, size=self.n)
        return (self.x, y)

    def fit_slope(self, x, y):
        """Fit a linear regression to x and y and return the estimated slope coefficient."""
        reg = LinearRegression()
        fit = reg.fit(x.reshape(-1, 1), y)
        return (fit.coef_[0])

    def run_simulations(self, n_sim):
        """Run n_sim simulations and store the estimated slopes in the slopes attribute."""
        self.slopes = [] 
        for i in range(n_sim):
            x,y = self.generate_data()
            self.slopes.append(self.fit_slope(x, y))

    def plot_sampling_distribution(self):
        """Plot a histogram of the simulated slopes to visualize the sampling distribution."""
        if len(self.slopes) == 0:
             print("Warning: run_simulations() must be called first to generate data")
        else:
          return(
              ggplot(pd.DataFrame({'slope': self.slopes})) +
              aes(x='slope') +
              geom_histogram(bins=int(np.ceil(np.log2(len(self.slopes))) + 1),
                             fill='black', alpha=0.7) +
              labs(title = "Distribution of Simulated Slope Estimates")
          )
    
    def find_prob(self, value, sided):
        """Estimate the probability of observing a slope beyond the given value.
        Arguments:
            value: The value to compare slopes against.
            sided: One of 'above', 'below', or 'two-sided'.
        """
        if len(self.slopes) == 0:
            print("Warning: run_simulations() must be called first to generate data")
        else:
            if sided not in ("above", "below", "two-sided"):
                print("Warning: sided must be one of 'above', 'below', or 'two-sided")
            elif sided == "above":
                bool_prob = np.array(self.slopes) > value 
                return bool_prob.mean()
            elif sided == "below":
                bool_prob = np.array(self.slopes) < value
                return bool_prob.mean()
            elif sided == "two-sided":
                bool_prob = np.abs(np.array(self.slopes)) > np.abs(value)
                return bool_prob.mean()
