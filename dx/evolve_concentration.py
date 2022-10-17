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
plt.style.use('./misc/experiments.mplstyle')
plt.ioff()


# --- PREPARE DATA ---

# Model hyperparameters
N_C = 1
DT = 14

MODEL_DIR = f"dx/models/GDP_{DT:.0f}day_NC{N_C}_vb/"

CHECKPOINT = "trained"
FIG_DIR = MODEL_DIR + "figures/"
if not Path(FIG_DIR).exists():
    Path(FIG_DIR).mkdir(parents=True)

print("Configuration done.")

# --- PREPARE DATA ---

DATA_DIR = f"data/GDP/{DT:.0f}day/"

X = np.load(DATA_DIR + "X0_train.npy")
Y = np.load(DATA_DIR + "DX_train.npy")

Xws = X.copy()
Xws[:, 0] -= 360.
Xes = X.copy()
Xes[:, 0] += 360.

# Periodicising X0.
X = np.concatenate((X, Xes, Xws), axis=0)
Y = np.concatenate((Y, Y, Y), axis=0)

# =============================================================================
# # Stereographic projection of X0.
# NPS = ccrs.NorthPolarStereo()
# X = NPS.transform_points(ccrs.PlateCarree(), X[:, 0], X[:, 1])[:, :2]
# =============================================================================

Xscaler = Scaler(X)
Yscaler = Scaler(Y)
X_size = X.shape[0]

del X, Y

print("Data prepared.")


# --- BUILD MODEL ---

# Data attributes
O_SIZE = len(Yscaler.mean)

# Model hyperparameters
N_C = 32

DENSITY_PARAMS_SIZE = tfpl.MixtureSameFamily.params_size(
    N_C, component_params_size=tfpl.MultivariateNormalTriL.params_size(O_SIZE))


def dense_layer(N, activation):
    return tfkl.Dense(N, activation=activation)


def var_layer(N, activation):
    return tfpl.DenseFlipout(
        N,
        kernel_divergence_fn=(
            lambda q, p, ignore: kl.kl_divergence(q, p) / X_size),
        bias_divergence_fn=(
            lambda q, p, ignore: kl.kl_divergence(q, p) / X_size),
        activation=activation)


# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():
model = tf.keras.Sequential([
    var_layer(256, 'relu'),
    var_layer(256, 'relu'),
    var_layer(256, 'relu'),
    var_layer(256, 'relu'),
    var_layer(512, 'relu'),
    var_layer(512, 'relu'),
    var_layer(32 * 6, None),
    tfpl.MixtureSameFamily(32, tfpl.MultivariateNormalTriL(2))]
)


# Load weights
model.load_weights(MODEL_DIR + CHECKPOINT + "/weights")

print("Model loaded.")


# Plot summary statistic on cartopy plot.
RES = 2.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)

X0 = np.array([-72.55, 33.67])[None, :]
gm_ = model(Xscaler.standardise(X0))


def p_X1_given_X0(X1):
    return Yscaler.invert_standardisation_prob(
        np.exp(
            gm_.log_prob(
                Yscaler.standardise(X1 - X0))))


p_X1_given_X0 = grid.eval_on_grid(p_X1_given_X0)


plt.figure(figsize=(6, 3))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
# ax.gridlines(draw_labels=True, dms=True,
#              x_inline=False, y_inline=False)
sca = ax.pcolormesh(grid.vertices[..., 0], grid.vertices[..., 1],
                    np.log(p_X1_given_X0),
                    cmap=cmocean.cm.amp,
                    shading='flat',
                    transform=ccrs.PlateCarree(),
                    vmin=-100)
ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor=None,
               facecolor='k',
               linewidth=0.3)
plt.colorbar(sca, extend=None, shrink=0.8)
plt.tight_layout()
plt.show()
plt.savefig(FIG_DIR + "cond" + ".png", dpi=576)
