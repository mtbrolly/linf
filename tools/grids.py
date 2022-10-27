"""
Module for grid creation.
"""

import numpy as np
# import regionmask
# import shapely
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
        self.xlims = xlims
        self.ylims = ylims
        self.n_x = n_x
        self.n_y = n_y
        self.R = R

# =============================================================================
#     def generate_mask(self):
#         land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
#         land_poly = shapely.ops.unary_union(land.polygons)
#         points = self.centres.reshape((-1, 2)).tolist()
#         centres_shp = shapely.geometry.MultiPoint(points)
#         land = [centres_shp.geoms[i].intersects(land_poly)
#                 for i in range(len(centres_shp.geoms))]
#         self.land = np.array(land).reshape(self.centres.shape[:-1] + (1, ))
# =============================================================================

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
        self.mean_flat = np.zeros((self.centres[..., 0].size, 2))
        self.cov_flat = np.zeros((self.centres[..., 0].size, 2, 2))

    def fit(self, X0, DX):

        X0_lon_bin = np.digitize(X0[:, 0], self.vertices[0, :, 0])
        X0_lat_bin = np.digitize(X0[:, 1], self.vertices[:, 0, 1])
        X0_bin = (X0_lat_bin - 1) * self.centres.shape[1] + (X0_lon_bin - 1)

        # Deal with corner case of binning positions on the dateline.
        X0_bin[X0_bin == self.centres[..., 0].size] -= 1
        assert X0_bin.max() < self.centres[..., 0].size, "Bin error."

        # Record which bins displacements start from.
        self.X0_some = np.bincount(
            X0_bin, minlength=self.centres[..., 0].size) != 0

        # Calculate maximum likelihood estimates of parameters.
        for i in range(self.centres[..., 0].size):
            X0_ind = X0_bin == i
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
    """
    Class for creating a discrete-time Markov chain model of drifter position
    with states given by grid cells.
    """

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

        # Record which bins displacements start from.
        self.X0_some = np.bincount(
            X0_bin, minlength=self.transition_matrix.shape[0]) != 0
        self.X0_some_2d = self.X0_some.reshape(self.centres.shape[:-1])

        # Compute maximum likelihood estimates of P_ij.
        for i in range(self.transition_matrix.shape[0]):
            X0_in_bin_i = X0_bin == i
            X1_counts = np.bincount(
                X1_bin[X0_in_bin_i], minlength=self.transition_matrix.shape[0])
            self.transition_matrix[i, :] = X1_counts / X0_in_bin_i.sum()

        self.transition_matrix_4d = self.transition_matrix.reshape(
            self.centres.shape[:-1] * 2)  # Right but with unusual indexing.
        self.reduced_transition_matrix = self.transition_matrix[
            :, self.X0_some][self.X0_some, :]  # !!! Wrong.

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

        X0_some = np.bincount(
            X0_bin, minlength=self.transition_matrix.shape[0]) != 0

        if np.logical_and(X0_some, 1 - self.X0_some).sum() > 0:
            print("Some initial positions in test data are outside the range "
                  + "of training data and will be ignored.")
        elif (self.transition_matrix[X0_bin, X1_bin] == 0.).sum() > 0:
            print("Some displacements in test data were not seen in training, "
                  + "and will be assigned probability zero; we exclude these "
                  + "from the mean log likelihood calculation.")

        mean_log_likelihood = np.nanmean(
            np.log(self.transition_matrix[X0_bin, X1_bin]
                   / self.areas.flatten()[X0_bin]
                   / self.areas.flatten()[X1_bin]),
            # where=(self.transition_matrix[X0_bin, X1_bin] != 0.)
            )

        return mean_log_likelihood
