"""
Utility functions for analysis of MDN models.
"""

import sys
import os
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cmocean
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import grids  # noqa: E402
plt.ioff()
plt.rcParams['figure.dpi'] = 150
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 12

tfkl = tf.keras.layers
tfpl = tfp.layers
tf.keras.backend.set_floatx("float64")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_mdn(DT, N_C):
    """
    Loads MDN model.
    """

    model = tf.keras.Sequential(
        [tfkl.Dense(256, activation='tanh'),
         tfkl.Dense(256, activation='tanh'),
         tfkl.Dense(256, activation='tanh'),
         tfkl.Dense(256, activation='tanh'),
         tfkl.Dense(512, activation='tanh'),
         tfkl.Dense(512, activation='tanh'),
         tfkl.Dense(N_C * 6, activation=None),
         tfpl.MixtureSameFamily(N_C, tfpl.MultivariateNormalTriL(2))])

    model.load_weights(
        f"dx/models/GDP_{DT:.0f}day_NC{N_C}/trained/weights").expect_partial()
    return model


def load_scalers(DT, N_C):
    """
    Loads scaler objects relating to MDN models.
    """

    with open(f"dx/models/GDP_{DT:.0f}day_NC{N_C}/Xscaler.pkl", "rb") as file:
        Xscaler = pickle.load(file)

    with open(f"dx/models/GDP_{DT:.0f}day_NC{N_C}/Yscaler.pkl", "rb") as file:
        Yscaler = pickle.load(file)
    return Xscaler, Yscaler


def mdn_mean_log_likelihood(X0val, DXval, DT, N_C, block_size=20000):
    """
    Computes the mean log likelihood of data under the MDN model.
    """

    model = load_mdn(DT=DT, N_C=N_C)
    Xscaler, Yscaler = load_scalers(DT=DT, N_C=N_C)

    def mll(X0val, DXval):
        gm_ = model(Xscaler.standardise(X0val))
        mean_log_likelihood = np.log(
            Yscaler.invert_standardisation_prob(
                np.exp(gm_.log_prob(Yscaler.standardise(DXval))))).mean()
        return mean_log_likelihood

    mlls = []
    for i in range(int(np.ceil(X0val.shape[0] / block_size))):
        mlls.append(mll(X0val[i * block_size: (i + 1) * block_size, :],
                        DXval[i * block_size: (i + 1) * block_size, :]))
        print('mll of block calculated')
    mean_log_likelihood = np.mean(np.array(mlls))
    return mean_log_likelihood


def plot_transition_density(X0, DT=4, N_C=32, res=2., radius=30.):
    """
    Produces a plot of the transition density (under an MDN model) given a
    certain initial position.
    """

    X0 = np.array(X0)[None, :]
    model = load_mdn(DT=DT, N_C=N_C)
    Xscaler, Yscaler = load_scalers(DT=DT, N_C=N_C)

    grid = grids.LonlatGrid(n_x=360 * res, n_y=180 * res)
    lims = [X0[0, 0] - radius, X0[0, 0] + radius,
            X0[0, 1] - radius, X0[0, 1] + radius]

    gm_ = model(Xscaler.standardise(X0))

    def p_X1_given_X0(X1):
        """
        Evaluates transition density for fixed X_0.
        """
        return Yscaler.invert_standardisation_prob(
            np.exp(
                gm_.log_prob(
                    Yscaler.standardise(X1 - X0))))

    p_X1_given_X0 = grid.eval_on_grid(p_X1_given_X0)

    with np.errstate(divide='ignore', invalid='ignore'):
        pc_data = np.log(p_X1_given_X0)

    plt.figure()
    ax = plt.axes(
        projection=cartopy.crs.PlateCarree(central_longitude=X0[0, 0]))
    ax.set_extent(lims, crs=cartopy.crs.PlateCarree())

    sca = ax.contourf(
        grid.centres[..., 0], grid.centres[..., 1], pc_data,
        levels=np.linspace(
            np.ma.masked_invalid(pc_data).min()
            + 0.96
            * (np.nanmax(pc_data) - np.ma.masked_invalid(pc_data).min()),
            np.nanmax(pc_data), 10),
        cmap=cmocean.cm.amp,
        transform=cartopy.crs.PlateCarree())

    ax.plot(X0[0, 0], X0[0, 1], 'yo', markersize=3.,
            transform=cartopy.crs.PlateCarree())

    ax.add_feature(cartopy.feature.NaturalEarthFeature(
        "physical", "land", "50m"),
        facecolor='k', edgecolor=None, zorder=100)
    plt.colorbar(sca, extend='min')
    plt.tight_layout()
    return
