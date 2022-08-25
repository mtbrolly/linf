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
                 xlims=(-180., 180.), ylims=(-90., 90.)):

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
        self.xlims = xlims
        self.ylims = ylims
        self.n_x = n_x
        self.n_y = n_y

    # def eval_on_grid(self, function, position='centres', scaler=None):
    #     """
    #     Evaluate a function on grid.
    #     """
    #     if position == 'centres':
    #         points = self.centres
    #     else:
    #         points = self.vertices

    #     points = np.reshape(points, (-1, 2))
    #     if scaler:
    #         points = scaler(points)
    #     f_evals = function(points)
    #     return f_evals

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

    def grid_points(self, points):
        """
        Get grid indices for points.
        """
        rescaled_points = points.copy()
        rescaled_points[..., 0] = ((rescaled_points[..., 0] - self.xlims[0]
                                    - 1e-8)
                                   / (self.xlims[1] - self.xlims[0])
                                   * self.n_x)
        rescaled_points[..., 1] = ((rescaled_points[..., 1] - self.ylims[0]
                                    - 1e-8)
                                   / (self.ylims[1] - self.ylims[0])
                                   * self.n_y)
        grid_indices = rescaled_points.astype(int)
        return grid_indices

    def bin_values(self, points, values):
        """
        Sort observation values by grid cell.
        """
        # Record initial shapes
        points_shape = points.shape[:-1]
        values_shape = values.shape[len(points_shape):]
        # Flatten points shape
        points = points.reshape((-1, 2))
        values = values.reshape((-1,) + values_shape)
        # Get indices
        grid_indices = self.grid_points(points)
        # Create array of empty lists in which to dump observations.
        bins = np.empty(self.centres[..., 0].T.shape, dtype=list)
        bins.fill([])
        for i in range(points.shape[0]):
            # TODO: tidy this.
            temp = bins[tuple(grid_indices[i, :])].copy()
            temp.append(values[i, ...])
            bins[tuple(grid_indices[i, :])] = temp.copy()
            # bins[tuple(grid_indices[i, :])].append(values[i, ...])
        return bins


class GriddedGaussianModel(LonlatGrid):
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

    def fit(self, X, Y, threshold_count=3):
        binned_values = self.bin_values(X, Y)
        for x in range(self.n_x):
            for y in range(self.n_y):
                bv = binned_values[x, y]
                count = len(bv)
                self.count[x, y] = count
                if count >= threshold_count:
                    self.mean[x, y, :] = np.mean(np.array(bv), axis=0)
                    self.cov[x, y, ...] = np.cov(np.array(bv), rowvar=False)
                else:
                    self.mean[x, y, :] = np.nan
                    self.cov[x, y, ...] = np.nan
        if (self.count < threshold_count).any():
            global_mean = np.mean(Y, axis=0)
            global_cov = np.cov(Y, rowvar=False)
            self.mean[..., 0] = np.nan_to_num(self.mean[..., 0],
                                              nan=global_mean[0])
            self.mean[..., 1] = np.nan_to_num(self.mean[..., 1],
                                              nan=global_mean[1])
            self.cov[..., 0, 0] = np.nan_to_num(self.cov[..., 0, 0],
                                                nan=global_cov[0, 0])
            self.cov[..., 0, 1] = np.nan_to_num(self.cov[..., 0, 1],
                                                nan=global_cov[0, 1])
            self.cov[..., 1, 0] = np.nan_to_num(self.cov[..., 1, 0],
                                                nan=global_cov[1, 0])
            self.cov[..., 1, 1] = np.nan_to_num(self.cov[..., 1, 1],
                                                nan=global_cov[1, 1])

    def log_likelihood(self, X, Y):
        grid_indices = self.grid_points(X)
        log_likelihood = 0.
        for i in range(X.shape[0]):
            log_likelihood += mvn.logpdf(
                Y[i, :], mean=self.mean[tuple(grid_indices[i, :])],
                cov=self.cov[tuple(grid_indices[i, :])])
        return log_likelihood

    def log_likelihood_based_on_global_averages(self, Y):
        global_mean = np.mean(Y, axis=0)
        global_cov = np.cov(Y, rowvar=False)
        log_likelihood = mvn.logpdf(Y, mean=global_mean, cov=global_cov).sum()
        return log_likelihood


def lonlat_to_cart(lonlatgrid):
    """
    (NO LONGER NEEDED)

    Subclass for converting a longitude-latitude grid to R^3 cartesian
    coordinates.
    """
    R = 1.  # 6371.
    lon_centres, lat_centres = lonlatgrid.centres
    x_centres = R * np.cos(lat_centres) * np.cos(lon_centres)
    y_centres = R * np.cos(lat_centres) * np.sin(lon_centres)
    z_centres = R * np.sin(lat_centres)
    lon_edges, lat_edges = lonlatgrid.vertices
    x_edges = R * np.cos(lat_edges) * np.cos(lon_edges)
    y_edges = R * np.cos(lat_edges) * np.sin(lon_edges)
    z_edges = R * np.sin(lat_edges)

    class cart_gridpoints(LonlatGrid):
        def __init__(self):
            self.centres = (x_centres, y_centres, z_centres)
            self.vertices = (x_edges, y_edges, z_edges)

    return cart_gridpoints()
