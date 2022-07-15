"""
Data preprocessing tools.
"""

import numpy as np


class Scaler:
    """
    Class for rescaling data, either by standardisation or normalisation.
    Includes methods for undoing the scaling for Gaussian mixture density
    networks.
    """

    def __init__(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        self._make_stats_array()

    def _make_stats_array(self):
        """
        This enforces compatibility for scalar variables.
        """
        if type(self.min) == np.float32:  # !!!
            self.min = np.array(self.min)[None]
            self.max = np.array(self.max)[None]
            self.mean = np.array(self.mean)[None]
            self.std = np.array(self.std)[None]

    def standardise(self, X):
        return (X - self.mean) / self.std

    def invert_standardisation(self, X):
        return (X * self.std) + self.mean

    def invert_standardisation_prob(self, prob):
        return prob / self.std.prod()

    def invert_standardisation_loc(self, loc):
        return self.invert_standardisation(loc)

    def invert_standardisation_cov(self, cov):
        return cov * (self.std[:, None] @ self.std[None, :])  # TODO: check.

    def normalise(self, X, feature_range=(-1.0, 1.0)):
        self.feature_range = feature_range
        return (feature_range[1] - feature_range[0]) * (X - self.min) / (
            self.max - self.min
        ) + feature_range[0]

    def invert_normalisation(self, X):
        return (X - self.feature_range[0]) / (
            self.feature_range[1] - self.feature_range[0]
        ) * (self.max - self.min) + self.min

    def invert_normalisation_prob(self, prob):
        factor = 1 / (
            (self.feature_range[1] - self.feature_range[0])
            / (self.max - self.min)
        )
        self.factor = factor  # TODO: check.
        return prob / factor

    def invert_normalisation_loc(self, loc):
        return self.invert_normalisation(loc)

    def invert_normalisation_cov(self, cov):
        factor = 1 / (
            (self.feature_range[1] - self.feature_range[0])
            / (self.max - self.min)
        )
        return cov * (factor[:, None] @ factor[None, :])
