"""
Script for analysis of, and figures relating to, dx models.
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
    MODEL_DIR = "dx/models/GDP_2day_vb_flipout_periodic/"
else:
    MODEL_DIR = "dx/models/GDP_4day_vb_flipout_periodic/"

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

# =============================================================================
# # Stereographic projection of X0.
# NPS = ccrs.NorthPolarStereo()
# X = NPS.transform_points(ccrs.PlateCarree(), X[:, 0], X[:, 1])[:, :2]
# =============================================================================

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

# Load log
history = pd.read_csv(MODEL_DIR + "log.csv")


# --- LOSS PLOT ---

plt.figure()
plt.plot(range(1, len(history) + 1), history['loss'], 'k',
         label='Training loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(FIG_DIR + "loss.png")
plt.close()


# Plot summary statistic on cartopy plot.
RES = 4.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)

gms_ = grid.eval_on_grid(model, scaler=Xscaler.standardise)

# =============================================================================
# def transform(X):
#     return Xscaler.standardise(
#         NPS.transform_points(
#             ccrs.PlateCarree(), X[..., 0], X[..., 1])[..., :2])
# gms_ = grid.eval_on_grid(model, scaler=transform)
# =============================================================================

means = []
covs = []
for i in range(10):
    gms_ = grid.eval_on_grid(model, scaler=Xscaler.standardise)
    means.append(
        Yscaler.invert_standardisation_loc(gms_.mean())[None, ...])
    covs.append(
        Yscaler.invert_standardisation_cov(gms_.covariance())[None, ...])

means = tf.concat(means, axis=0)
covs = tf.concat(covs, axis=0)

mean_of_mean = tf.math.reduce_mean(means, axis=0)
mean_of_cov = tf.math.reduce_mean(covs, axis=0)
std_of_mean = tf.math.reduce_std(means, axis=0)
std_of_cov = tf.math.reduce_std(covs, axis=0)


fig_names = ["mean_dx", "mean_dx_m", "mean_dy", "mean_dy_m",
             "var_dx", "var_dx_m", "diff_x",
             "cov_dx_dy", "cov_dx_dy_m", "diff_xy",
             "var_dy", "var_dy_m", "diff_y",
             "cv_diff_x"]
cmaps = [cmocean.cm.delta, cmocean.cm.amp, cmocean.cm.matter,
         cmocean.cm.balance]
lon_deg_to_m = grid.R * np.deg2rad(1) * np.cos(np.deg2rad(
    grid.centres[..., 1]))
lat_deg_to_m = grid.R * np.deg2rad(1)

mean = mean_of_mean
cov = mean_of_cov

for i in range(13):
    if i == 0:
        pc_data = mean[..., 0].numpy()
        cmap = cmaps[0]
        # LIM = max((-pc_data.min(), pc_data.max()))
        LIM = 1.5
        CLIM = [-LIM, LIM]
        NORM = None
        EXTEND = 'both'
    elif i == 1:
        pc_data = mean[..., 0].numpy()
        pc_data *= lon_deg_to_m
        cmap = cmaps[0]
        LIM = 100000.
        CLIM = [-LIM, LIM]
        EXTEND = 'both'
    elif i == 2:
        pc_data = mean[..., 1].numpy()
        cmap = cmaps[0]
        LIM = 1.5
        CLIM = [-LIM, LIM]
        EXTEND = 'both'
    elif i == 3:
        pc_data = mean[..., 1].numpy()
        pc_data *= lat_deg_to_m
        cmap = cmaps[0]
        LIM = 100000.
        CLIM = [-LIM, LIM]
        EXTEND = 'both'
    elif i == 4:
        pc_data = cov[..., 0, 0].numpy()
        cmap = cmaps[1]
        CLIM = [0., 1.]
        EXTEND = 'max'
    elif i == 5:
        pc_data = cov[..., 0, 0].numpy()
        pc_data *= lon_deg_to_m ** 2
        cmap = cmaps[1]
        LIM = 1e10
        CLIM = [0., LIM]
        EXTEND = 'max'
    elif i == 6:
        pc_data = cov[..., 0, 0].numpy()
        pc_data *= lon_deg_to_m ** 2 / (DT * 24 * 3600) / 2
        cmap = cmaps[1]
        NORM = colors.LogNorm(1e3, 3e4)
        CLIM = None
        EXTEND = 'both'
    elif i == 7:
        pc_data = cov[..., 0, 1].numpy()
        cmap = cmaps[0]
        LIM = max((-pc_data.min(), pc_data.max()))
        LIM = 0.3
        CLIM = [-LIM, LIM]
        NORM = None
        EXTEND = 'both'
    elif i == 8:
        pc_data = cov[..., 0, 1].numpy()
        pc_data *= lon_deg_to_m * lat_deg_to_m
        cmap = cmaps[0]
        LIM = 5e9
        CLIM = [-LIM, LIM]
        NORM = None
        EXTEND = 'both'
    elif i == 9:
        pc_data = cov[..., 0, 1].numpy()
        pc_data *= lon_deg_to_m * lat_deg_to_m / (DT * 24 * 3600) / 2
        cmap = cmaps[0]
        LIM = 1e4
        CLIM = [-LIM, LIM]
        NORM = None
        EXTEND = 'both'
    elif i == 10:
        pc_data = cov[..., 1, 1].numpy()
        cmap = cmaps[1]
        CLIM = [0., 0.6]
        NORM = None
        EXTEND = 'max'
    elif i == 11:
        pc_data = cov[..., 1, 1].numpy()
        pc_data *= lat_deg_to_m ** 2
        cmap = cmaps[1]
        LIM = 1e10
        CLIM = [0., LIM]
        NORM = None
        EXTEND = 'max'
    elif i == 12:
        pc_data = cov[..., 1, 1].numpy()
        pc_data *= lat_deg_to_m ** 2 / (DT * 24 * 3600) / 2
        cmap = cmaps[1]
        NORM = colors.LogNorm(1e3, 3e4)
        CLIM = None
        EXTEND = 'both'
    elif i == 13:
        pc_data_std = std_of_cov[..., 0, 0].numpy()
        pc_data = cov[..., 0, 0].numpy()
        pc_data = pc_data_std / pc_data
        cmap = 'jet'  # cmaps[1]
        NORM = colors.LogNorm(1e-2, 1e0)
        # NORM = None
        # CLIM = [1e-2, 1e0]
        EXTEND = 'both'

    plt.figure(figsize=(6, 3))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
    ax.spines['geo'].set_visible(False)
    # ax.gridlines(draw_labels=True, dms=True,
    #              x_inline=False, y_inline=False)
    sca = ax.pcolormesh(grid.vertices[..., 0], grid.vertices[..., 1],
                        pc_data,
                        cmap=cmap,
                        norm=NORM,
                        clim=CLIM,
                        shading='flat',
                        transform=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor=None,
                   facecolor='k',
                   linewidth=0.3)
    plt.colorbar(sca, extend=EXTEND, shrink=0.8,
                 # pad=0.05, orientation='horizontal', fraction=0.05
                 )
    plt.tight_layout()

    # plt.show()
    plt.savefig(FIG_DIR + fig_names[i] + ".png")
    plt.close()
