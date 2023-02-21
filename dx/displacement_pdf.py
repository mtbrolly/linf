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


# Model hyperparameters
N_C = 32
DT = 4

MODEL_DIR = (f"dx/models/GDP_{DT:.0f}day_NC{N_C}")

CHECKPOINT = "trained"

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

Xscaler = Scaler(X)
Yscaler = Scaler(Y)
X_size = X.shape[0]

del X, Y, Xws, Xes

print("Data prepared.")


# --- BUILD MODEL ---

# Data attributes
O_SIZE = len(Yscaler.mean)


DENSITY_PARAMS_SIZE = tfpl.MixtureSameFamily.params_size(
    N_C, component_params_size=tfpl.MultivariateNormalTriL.params_size(O_SIZE))


def dense_layer(N, activation):
    return tfkl.Dense(N, activation=activation)


activation_fn = 'tanh'

model = tf.keras.Sequential([
    dense_layer(256, activation_fn),
    dense_layer(256, activation_fn),
    dense_layer(256, activation_fn),
    dense_layer(256, activation_fn),
    dense_layer(512, activation_fn),
    dense_layer(512, activation_fn),
    dense_layer(N_C * 6, None),
    tfpl.MixtureSameFamily(N_C, tfpl.MultivariateNormalTriL(2))]
)


# Load weights
model.load_weights(MODEL_DIR + CHECKPOINT + "/weights")

print("Model loaded.")


# Plot summary statistic on cartopy plot.
RES = 2.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)

# X0 = np.array([-72.55, 33.67])[None, :]
X0 = np.array([-74.5, 34.85])[None, :]
lims = [-90, -55, 20, 50]

# X0 = np.array([-50, 7.3])[None, :]
# lims = [-65, -20, -5, 25]

# X0 = np.array([6.9, 54.4])[None, :]
# lims = [-15, 10, 45, 65]

gm_ = model(Xscaler.standardise(X0))


def p_X1_given_X0(X1):
    return Yscaler.invert_standardisation_prob(
        np.exp(
            gm_.log_prob(
                Yscaler.standardise(X1 - X0))))


p_X1_given_X0 = grid.eval_on_grid(p_X1_given_X0)

pc_data = np.log(p_X1_given_X0)

plt.figure(figsize=(4, 3))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.))
ax.set_extent(lims, crs=ccrs.PlateCarree())

sca = ax.contourf(grid.centres[..., 0], grid.centres[..., 1],
                  pc_data,
                  levels=np.linspace(-20., pc_data.max(), 10),
                  cmap=cmocean.cm.amp,
                  transform=ccrs.PlateCarree())

ax.plot(X0[0, 0], X0[0, 1], 'yo', markersize=3.)

# ax.add_feature(cartopy.feature.NaturalEarthFeature(
#     "physical", "land", "50m"),
#     facecolor='k', edgecolor=None, zorder=100)
ax.coastlines()
plt.colorbar(sca, extend='min')
plt.tight_layout()
# plt.show()
plt.savefig(MODEL_DIR + "figures/cond_gulf_stream2.png")
# plt.savefig(MODEL_DIR + "figures/cond_brazil.png")
plt.close()
