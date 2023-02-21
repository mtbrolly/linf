"""
Data preprocessing tools.
"""

import numpy as np


class Scaler:
    """
    Class for scaling data by standardisation. Includes methods for inverting
    the scaling of data and related probability densities, means and
    covariances.
    """

    def __init__(self, X):
        assert len(X.shape) > 1, "X must have dimension greater than 1."
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def standardise(self, X):
        return (X - self.mean) / self.std

    def invert_standardisation(self, X):
        return (X * self.std) + self.mean

    def invert_standardisation_prob(self, prob):
        return prob / self.std.prod()

    def invert_standardisation_log_prob(self, prob):
        return prob - np.log(self.std.prod())

    def invert_standardisation_loc(self, loc):
        return self.invert_standardisation(loc)

    def invert_standardisation_cov(self, cov):
        return cov * (self.std[:, None] @ self.std[None, :])
