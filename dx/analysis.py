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
tf.keras.backend.set_floatx("float64")
plt.style.use('./misc/paper.mplstyle')
plt.ioff()

DT = 4

assert DT in (2, 4), "No model for this value of DT."

if DT == 2:
    MODEL_DIR = "dx/models/GDP_2day_ml_periodic/"
else:
    MODEL_DIR = "dx/models/GDP_4day_ml_periodic/"

CHECKPOINT = "trained"
# CHECKPOINT = "checkpoint_epoch_01"
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

# Load log
history = pd.read_csv(MODEL_DIR + "log.csv")


# --- LOSS PLOT ---

plt.figure()
plt.plot(range(1, len(history) + 1), history['loss'], 'k',
         label='Training loss')
plt.plot(range(1, len(history) + 1), history['val_loss'], 'grey',
         label='Test loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(FIG_DIR + "loss.png")
plt.close()


# Plot summary statistic on cartopy plot.
RES = 3.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)

gms_ = grid.eval_on_grid(model, scaler=Xscaler.standardise)

# =============================================================================
# def transform(X):
#     return Xscaler.standardise(
#         NPS.transform_points(
#             ccrs.PlateCarree(), X[..., 0], X[..., 1])[..., :2])
# gms_ = grid.eval_on_grid(model, scaler=transform)
# =============================================================================

mean = Yscaler.invert_standardisation_loc(gms_.mean())
cov = Yscaler.invert_standardisation_cov(gms_.covariance())

fig_names = ["mean_dx", "mean_dx_m", "mean_dy", "mean_dy_m",
             "var_dx", "var_dx_m", "diff_x",
             "cov_dx_dy", "cov_dx_dy_m", "diff_xy",
             "var_dy", "var_dy_m", "diff_y"]
cmaps = [cmocean.cm.delta, cmocean.cm.amp, cmocean.cm.matter,
         cmocean.cm.balance]
R = 6378137.  # Equatorial radius in meters.
lon_deg_to_m = R * np.deg2rad(1) * np.cos(np.deg2rad(grid.centres[..., 1]))
lat_deg_to_m = R * np.deg2rad(1)

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
        # LIM = max((-pc_data.min(), pc_data.max()))
        LIM = 100000.
        CLIM = [-LIM, LIM]
        EXTEND = 'both'
    elif i == 2:
        pc_data = mean[..., 1].numpy()
        cmap = cmaps[0]
        # LIM = max((-pc_data.min(), pc_data.max()))
        LIM = 1.5
        CLIM = [-LIM, LIM]
        EXTEND = 'both'
    elif i == 3:
        pc_data = mean[..., 1].numpy()
        pc_data *= lat_deg_to_m
        cmap = cmaps[0]
        # LIM = max((-pc_data.min(), pc_data.max()))
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
        # LIM = pc_data.max()
        LIM = 1e10
        CLIM = [0., LIM]
        EXTEND = 'max'
    elif i == 6:
        pc_data = cov[..., 0, 0].numpy()
        pc_data *= lon_deg_to_m ** 2 / (DT * 24 * 3600) / 2
        cmap = cmaps[1]
        # LIM = pc_data.max()
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
        # LIM = max((-pc_data.min(), pc_data.max()))
        LIM = 5e9
        CLIM = [-LIM, LIM]
        NORM = None
        EXTEND = 'both'
    elif i == 9:
        pc_data = cov[..., 0, 1].numpy()
        pc_data *= lon_deg_to_m * lat_deg_to_m / (DT * 24 * 3600) / 2
        cmap = cmaps[0]
        # LIM = max((-pc_data.min(), pc_data.max()))
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
        # LIM = pc_data.max()
        LIM = 1e10
        CLIM = [0., LIM]
        NORM = None
        EXTEND = 'max'
    elif i == 12:
        pc_data = cov[..., 1, 1].numpy()
        pc_data *= lat_deg_to_m ** 2 / (DT * 24 * 3600) / 2
        cmap = cmaps[1]
        # LIM = pc_data.max()
        NORM = colors.LogNorm(1e3, 3e4)
        CLIM = None
        EXTEND = 'both'

    plt.figure(figsize=(6, 3))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
    ax.spines['geo'].set_visible(False)
    # ax.gridlines(draw_labels=True, dms=True,
    #              x_inline=False, y_inline=False)
    # plt.title(fig_titles[i])
    sca = ax.pcolormesh(grid.vertices[..., 0], grid.vertices[..., 1],
                        pc_data,
                        cmap=cmap,
                        norm=NORM,
                        clim=CLIM,
                        shading='flat',
                        transform=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor=None,
                   # facecolor=cartopy.feature.COLORS['land_alt1'],
                   facecolor='k',
                   linewidth=0.3)
    plt.colorbar(sca, extend=EXTEND, shrink=0.8,
                 # pad=0.05, orientation='horizontal', fraction=0.05
                 )
    plt.tight_layout()

    # plt.show()
    plt.savefig(FIG_DIR + fig_names[i] + ".png")
    plt.close()
