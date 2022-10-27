"""
Script for applying Chapman-Kolmogorov equation on a lon-lat grid.
"""

from pathlib import Path
import cmocean
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy
from tools.preprocessing import Scaler
from tools import grids

ccrs = cartopy.crs
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
kl = tfd.kullback_leibler
tf.keras.backend.set_floatx("float64")
plt.style.use('./misc/paper.mplstyle')
plt.ioff()

DT = 2

assert DT in (2, 4), "No model for this value of DT."

if DT == 2:
    MODEL_DIR = "dx/models/GDP_2day_ml_periodic/"
else:
    MODEL_DIR = "dx/models/GDP_4day_ml_periodic/"

CHECKPOINT = "trained"
FIG_DIR = MODEL_DIR + "figures/"
if not Path(FIG_DIR).exists():
    Path(FIG_DIR).mkdir(parents=True)


# --- PREPARE DATA ---

if DT == 2:
    DATA_DIR = "data/GDP/2day/"
else:
    DATA_DIR = "data/GDP/4day/"

X = np.load(DATA_DIR + "X0_train.npy")
Y = np.load(DATA_DIR + "DX_train.npy")

Xws = X.copy()
Xws[:, 0] -= 360.
Xes = X.copy()
Xes[:, 0] += 360.

# Periodicising X0.
X = np.concatenate((X, Xes, Xws), axis=0)
Y = np.concatenate((Y, Y, Y), axis=0)

Xscaler = Scaler(X)
Yscaler = Scaler(Y)
X_size = X.shape[0]

del X, Y


# --- BUILD MODEL ---

# Data attributes
O_SIZE = len(Yscaler.mean)

# Model hyperparameters
N_C = 32

DENSITY_PARAMS_SIZE = tfpl.MixtureSameFamily.params_size(
    N_C, component_params_size=tfpl.MultivariateNormalTriL.params_size(O_SIZE))


# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():
model = tf.keras.Sequential([
    tfkl.Dense(256, activation='relu'),
    tfkl.Dense(256, activation='relu'),
    tfkl.Dense(256, activation='relu'),
    tfkl.Dense(256, activation='relu'),
    tfkl.Dense(512, activation='relu'),
    tfkl.Dense(512, activation='relu'),
    tfkl.Dense(DENSITY_PARAMS_SIZE),
    tfpl.MixtureSameFamily(N_C, tfpl.MultivariateNormalTriL(O_SIZE))]
)

# Load weights
model.load_weights(MODEL_DIR + CHECKPOINT + "/weights")


# Plot summary statistic on cartopy plot.
RES = 4.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)

X0 = np.array([-72.55, 33.67])[None, :]
gm_ = model(Xscaler.standardise(X0))

gm_comp_weights = tf.math.softmax(
    gm_.parameters['mixture_distribution'].logits).numpy() > 0.05  # TODO: check

gm_comp_means = Yscaler.invert_standardisation_loc(
    gm_.parameters['components_distribution'].mean()).numpy().squeeze() + X0

gm_comp_covs = Yscaler.invert_standardisation_cov(
    gm_.parameters['components_distribution'].covariance()).numpy().squeeze()

eig = np.linalg.eig(gm_comp_covs[0, ...])

gm_comp_means = gm_comp_means[gm_comp_weights.flatten(), :]


def p_X1_given_X0(X1):
    return Yscaler.invert_standardisation_prob(
        np.exp(
            gm_.log_prob(
                Yscaler.standardise(X1 - X0))))


p_X1_given_X0 = grid.eval_on_grid(p_X1_given_X0)


plt.figure(figsize=(6, 3))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
# ax.spines['geo'].set_visible(False)
# ax.gridlines(draw_labels=True, dms=True,
#              x_inline=False, y_inline=False)
ax.scatter(X0[0, 0], X0[0, 1], c='b', s=1, marker='x',
           transform=ccrs.PlateCarree(), zorder=10)
for i in range(len(gm_comp_means)):
    ax.scatter(gm_comp_means[i, 0], gm_comp_means[i, 1],
               c='grey', s=1, marker='x',
               transform=ccrs.PlateCarree(), zorder=10)
    vals, vecs = np.linalg.eig(gm_comp_covs[i, ...])
    # vals *= 10
    ax.arrow(gm_comp_means[i, 0], gm_comp_means[i, 1],
             vals[0] * vecs[0, 0], vals[0] * vecs[1, 0],
             color='k', transform=ccrs.PlateCarree(), zorder=100)
    ax.arrow(gm_comp_means[i, 0], gm_comp_means[i, 1],
             vals[1] * vecs[0, 1], vals[1] * vecs[1, 1],
             color='k', transform=ccrs.PlateCarree(), zorder=100)
sca = ax.pcolormesh(grid.vertices[..., 0], grid.vertices[..., 1],
                    np.log(p_X1_given_X0),
                    cmap=cmocean.cm.amp,
                    shading='flat',
                    transform=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor=None,
               facecolor='k',
               linewidth=0.3)
plt.colorbar(sca, extend=None, shrink=0.8)
plt.tight_layout()
plt.show()
