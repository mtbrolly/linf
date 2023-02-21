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

MODEL_DIR = (f"dx/models/GDP_{DT:.0f}day_NC{N_C}"
             + "_ml_flipout_Adam_tanh_lr5em5_pat50_val20/")

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

m32 = tf.keras.Sequential([
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
m32.load_weights(MODEL_DIR + CHECKPOINT + "/weights")

print("Model with 32 components loaded.")


# ---------------------------------------------

# Model hyperparameters
N_C = 1
DT = 4

MODEL_DIR = (f"dx/models/GDP_{DT:.0f}day_NC{N_C}"
             + "_ml_flipout_Adam_tanh_lr5em5_pat50_val20/")

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

m1 = tf.keras.Sequential([
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
m1.load_weights(MODEL_DIR + CHECKPOINT + "/weights")

print("Model with 1 component loaded.")


# Plot summary statistic on cartopy plot.
RES = 1.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)


KL_X0 = np.zeros_like(grid.centres[..., 0])


for i in range(KL_X0.shape[0]):
    print(i)
    N = 1000
    X0 = grid.centres[i: i + 1]
    DX_samples = m32(Xscaler.standardise(X0)).sample(N)
    lp_m32 = Yscaler.invert_standardisation_log_prob(
        m32(Xscaler.standardise(X0)).log_prob(DX_samples).numpy())
    lp_m1 = Yscaler.invert_standardisation_log_prob(
        m1(Xscaler.standardise(X0)).log_prob(DX_samples).numpy())
    KL_X0[i] = (lp_m32 - lp_m1).mean(axis=0)


np.save(MODEL_DIR + "KL.npy", KL_X0)

pc_data = KL_X0

# Repeated to set the directory for figures.
N_C = 32
DT = 4
MODEL_DIR = (f"dx/models/GDP_{DT:.0f}day_NC{N_C}"
             + "_ml_flipout_Adam_tanh_lr5em5_pat50_val20/")

plt.figure(figsize=(6, 3))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
ax.spines['geo'].set_visible(False)

sca = ax.pcolormesh(grid.vertices[..., 0], grid.vertices[..., 1],
                    pc_data,
                    # levels=np.linspace(-20., pc_data.max(), 10),
                    cmap='Oranges',  # cmocean.cm.amp,
                    vmin=0.,
                    vmax=.3,
                    transform=ccrs.PlateCarree())

# sca = ax.contour(grid.centres[..., 0], grid.centres[..., 1],
#                   pc_data,
#                   levels=np.linspace(0., 1, 10),
#                   cmap=cmocean.cm.amp,
#                   vmin=0.,
#                   vmax=1.,
#                   transform=ccrs.PlateCarree())

# ax.plot(X0[0, 0], X0[0, 1], 'yo', markersize=3.)

ax.add_feature(cartopy.feature.NaturalEarthFeature(
    "physical", "land", "50m"),
    facecolor='k', edgecolor=None, zorder=100)
plt.colorbar(sca, extend='max')
plt.tight_layout()
# plt.show()
plt.savefig(MODEL_DIR + "figures/KL.png")
plt.close()
