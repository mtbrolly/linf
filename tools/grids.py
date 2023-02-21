"""
Module for grid creation.
"""

import numpy as np
from scipy.stats import multivariate_normal as mvn


class LonlatGrid():
    """
    Class for longitude-latitude grids.
    """

    def __init__(self, n_x=360, n_y=180,
                 xlims=(-180., 180.), ylims=(-90., 90.),
                 R=6378137.):

        n_x = int(n_x)
        n_y = int(n_y)

        lon_edges = np.linspace(xlims[0], xlims[1], n_x + 1)
        lat_edges = np.linspace(ylims[0], ylims[1], n_y + 1)
        lon_middles = lon_edges[:-1] + (lon_edges[1] - lon_edges[0]) / 2
        lat_middles = lat_edges[:-1] + (lat_edges[1] - lat_edges[0]) / 2

        lon_vertices, lat_vertices = np.meshgrid(lon_edges, lat_edges)
        lon_centres, lat_centres = np.meshgrid(lon_middles, lat_middles)

        self.vertices = np.concatenate((lon_vertices[..., None],
                                        lat_vertices[..., None]), axis=2)
        self.centres = np.concatenate((lon_centres[..., None],
                                       lat_centres[..., None]), axis=2)
        self.areas = R ** 2 * (
            np.sin(np.deg2rad(self.vertices[1:, 1:, 1]))
            - np.sin(np.deg2rad(self.vertices[:-1, :-1, 1]))) * (
                np.deg2rad(
                    self.vertices[1:, 1:, 0] - self.vertices[:-1, :-1, 0]))
        self.deg_areas = (
            (self.vertices[1:, 1:, 0] - self.vertices[:-1, :-1, 0])
            * (self.vertices[1:, 1:, 1] - self.vertices[:-1, :-1, 1]))
        self.xlims = xlims
        self.ylims = ylims
        self.n_x = n_x
        self.n_y = n_y
        self.R = R

    def eval_on_grid(self, function, position='centres', scaler=None):
        """
        Evaluate a function on grid.
        """

        if position == 'centres':
            points = self.centres
        else:
            points = self.vertices

        if scaler:
            points = scaler(points)
        f_evals = function(points)
        return f_evals


class GTGP(LonlatGrid):
    """
    Class for creating a Markov process model of drifter dynamics with
    Gaussian transition probability whose mean and covariance are cell-wise
    constant on a grid.
    """

    def __init__(self, n_x=360, n_y=180, xlims=(-180., 180.),
                 ylims=(-90., 90.)):
        super().__init__(n_x=n_x, n_y=n_y,
                         xlims=xlims, ylims=ylims)
        self.count = np.zeros(self.centres.shape[:-1][::-1] + (1,), dtype=int)
        self.mean = np.zeros(self.centres.shape[:-1][::-1] + (2,))
        self.cov = np.zeros(self.centres.shape[:-1][::-1] + (2, 2))
        self.count_flat = np.zeros((self.centres[..., 0].size))
        self.mean_flat = np.zeros((self.centres[..., 0].size, 2))
        self.cov_flat = np.zeros((self.centres[..., 0].size, 2, 2))

    def count_X0s(self, X0, DX):
        X0_lon_bin = np.digitize(X0[:, 0], self.vertices[0, :, 0])
        X0_lat_bin = np.digitize(X0[:, 1], self.vertices[:, 0, 1])
        X0_bin = (X0_lat_bin - 1) * self.centres.shape[1] + (X0_lon_bin - 1)

        # Deal with corner case of binning positions on the dateline.
        X0_bin[X0_bin == self.centres[..., 0].size] -= 1
        assert X0_bin.max() < self.centres[..., 0].size, "Bin error."

        # Record which bins displacements start from.
        self.count = np.bincount(
            X0_bin, minlength=self.centres[..., 0].size
            ).reshape(self.count.shape, order='F')

    def fit(self, X0, DX):
        self.global_mean = np.mean(DX, axis=0)
        self.global_cov = np.cov(DX, rowvar=False)

        X0_lon_bin = np.digitize(X0[:, 0], self.vertices[0, :, 0])
        X0_lat_bin = np.digitize(X0[:, 1], self.vertices[:, 0, 1])
        X0_bin = (X0_lat_bin - 1) * self.centres.shape[1] + (X0_lon_bin - 1)

        # Deal with corner case of binning positions on the dateline.
        X0_bin[X0_bin == self.centres[..., 0].size] -= 1
        assert X0_bin.max() < self.centres[..., 0].size, "Bin error."

        # Record which bins displacements start from.
        self.count = np.bincount(
            X0_bin, minlength=self.centres[..., 0].size)
        self.count_flat = np.reshape(self.count, self.count_flat.shape)
        self.X0_some = self.count != 0

        # Calculate maximum likelihood estimates of parameters.
        for i in range(self.centres[..., 0].size):
            X0_ind = X0_bin == i
            if self.count_flat[i] < 3:
                self.mean_flat[i, :] = self.global_mean
                self.cov_flat[i, ...] = self.global_cov
            else:
                self.mean_flat[i, :] = np.mean(DX[X0_ind])
                self.cov_flat[i, ...] = np.cov(DX[X0_ind], rowvar=False)

        self.mean = self.mean_flat.reshape(self.mean.shape, order='F')
        self.cov = self.cov_flat.reshape(self.cov.shape, order='F')

    def log_likelihood(self, X0, DX):
        X0_lon_bin = np.digitize(X0[:, 0], self.vertices[0, :, 0])
        X0_lat_bin = np.digitize(X0[:, 1], self.vertices[:, 0, 1])
        X0_bin = (X0_lat_bin - 1) * self.centres.shape[1] + (X0_lon_bin - 1)
        means = self.mean_flat[X0_bin]
        covs = self.cov_flat[X0_bin]
        log_likelihood = 0.

        for i in range(DX.shape[0]):
            log_likelihood += mvn.logpdf(DX[i, :], mean=means[i, :],
                                         cov=covs[i, ...])
        return log_likelihood

    def mean_log_likelihood(self, X0, DX):
        return self.log_likelihood(X0, DX) / DX.shape[0]


class DTMC(LonlatGrid):
    def __init__(self, n_x=360, n_y=180, xlims=(-180., 180.),
                 ylims=(-90., 90.)):
        super().__init__(n_x=n_x, n_y=n_y,
                         xlims=xlims, ylims=ylims)
        self.transition_matrix = np.zeros((self.centres[..., 0].size, ) * 2)

    def fit(self, X0, DX):
        """
        Fit DTMC model given (X0, DX) pairs.
        """

        # Get X1 from DX.
        X1 = X0 + DX
        del DX
        X1[(X1 > 180)[:, 0], 0] -= 360.
        X1[(X1 < -180)[:, 0], 0] += 360.

        # Get bin indices for X0 and X1.
        X0_lon_bin = np.digitize(X0[:, 0], self.vertices[0, :, 0])
        X1_lon_bin = np.digitize(X1[:, 0], self.vertices[0, :, 0])
        X0_lat_bin = np.digitize(X0[:, 1], self.vertices[:, 0, 1])
        X1_lat_bin = np.digitize(X1[:, 1], self.vertices[:, 0, 1])

        X0_bin = (X0_lat_bin - 1) * self.centres.shape[1] + (X0_lon_bin - 1)
        X1_bin = (X1_lat_bin - 1) * self.centres.shape[1] + (X1_lon_bin - 1)

        # Deal with corner case of binning positions on the dateline.
        X0_bin[X0_bin == self.transition_matrix.shape[0]] -= 1
        X1_bin[X1_bin == self.transition_matrix.shape[0]] -= 1
        assert (
            X0_bin.max() < self.transition_matrix.shape[0]
            and X1_bin.max() < self.transition_matrix.shape[0]), "Bin error."

        self.X0_bin = X0_bin
        self.X1_bin = X1_bin

        # Record which bins displacements start from.
        self.X0_some = np.bincount(
            X0_bin, minlength=self.transition_matrix.shape[0]) != 0
        self.X0_some_2d = self.X0_some.reshape(self.centres.shape[:-1])  # 'xy'

        # Compute maximum likelihood estimates of P_ij.
        # Probability of going from cell i to cell j.
        for i in range(self.transition_matrix.shape[0]):
            # Find which start in cell i.
            X0_in_bin_i = X0_bin == i
            # Of those, how many go to each cell j.
            X1_counts = np.bincount(
                X1_bin[X0_in_bin_i], minlength=self.transition_matrix.shape[0])
            # Compute proportion that went to each j. Some rows may be nan.
            self.transition_matrix[i, :] = X1_counts / X0_in_bin_i.sum()
        assert np.all(np.abs(np.sum(
            self.transition_matrix[
                np.isnan(self.transition_matrix[:, 0]) == False],  # noqa: E712
            axis=1) - 1. < 1e-12)), "Transition matrix not row stochastic."

        self.transition_matrix_4d = self.transition_matrix.reshape(
            self.centres.shape[:-1] * 2)  # Right but with unusual indexing.

        self.reduced_transition_matrix = self.transition_matrix[
            :, self.X0_some][self.X0_some, :]

    def mean_log_likelihood(self, X0, DX):

        # Get X1 from DX.
        X1 = X0 + DX
        del DX
        X1[(X1 > 180)[:, 0], 0] -= 360.
        X1[(X1 < -180)[:, 0], 0] += 360.

        # Get bin indices for X0 and X1.
        X0_lon_bin = np.digitize(X0[:, 0], self.vertices[0, :, 0])
        X1_lon_bin = np.digitize(X1[:, 0], self.vertices[0, :, 0])
        X0_lat_bin = np.digitize(X0[:, 1], self.vertices[:, 0, 1])
        X1_lat_bin = np.digitize(X1[:, 1], self.vertices[:, 0, 1])

        X0_bin = (X0_lat_bin - 1) * self.centres.shape[1] + (X0_lon_bin - 1)
        X1_bin = (X1_lat_bin - 1) * self.centres.shape[1] + (X1_lon_bin - 1)

        # Deal with corner case of binning positions on the dateline.
        X0_bin[X0_bin == self.transition_matrix.shape[0]] -= 1
        X1_bin[X1_bin == self.transition_matrix.shape[0]] -= 1
        assert (
            X0_bin.max() < self.transition_matrix.shape[0]
            and X1_bin.max() < self.transition_matrix.shape[0]), "Bin error."

        mean_log_likelihood = np.mean(
            # np.ma.masked_invalid(  # Use to ignore -inf when taking mean.
            np.log(self.transition_matrix[X0_bin, X1_bin]
                   / self.deg_areas.flatten()[X1_bin])
            )

        return mean_log_likelihood
