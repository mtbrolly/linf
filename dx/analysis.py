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


MODEL_DIR = "dx/models/GDP_wrapped_2908/"
CHECKPOINT = "trained"
# CHECKPOINT = "checkpoint_epoch_01"
FIG_DIR = MODEL_DIR + "figuRES/"
if not Path(FIG_DIR).exists():
    Path(FIG_DIR).mkdir(parents=True)


# --- PREPARE DATA ---

DATA_DIR = "data/GDP/2day/"

X = np.load(DATA_DIR + "X0_train.npy")
Y = np.load(DATA_DIR + "DX_train.npy")

Xws = X.copy()
Xws[:, 0] -= 360.
Xes = X.copy()
Xes[:, 0] += 360.

# Periodicising X0.
X = np.concatenate((X, Xes, Xws), axis=0)
Y = np.concatenate((Y, Y, Y), axis=0)

# # Stereographic projection of X0.
# NPS = ccrs.NorthPolarStereo()
# X = NPS.transform_points(ccrs.PlateCarree(), X[:, 0], X[:, 1])[:, :2]

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
RES = 2.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)


# def transform(X):
#     return Xscaler.standardise(
#         NPS.transform_points(
#             ccrs.PlateCarree(), X[..., 0], X[..., 1])[..., :2])


gms_ = grid.eval_on_grid(model, scaler=Xscaler.standardise)
# gms_ = grid.eval_on_grid(model, scaler=transform)  # !!!
mean = Yscaler.invert_standardisation_loc(gms_.mean())
cov = Yscaler.invert_standardisation_cov(gms_.covariance())

fig_names = ["mean_dx", "mean_dy", "var_dx", "cov_dx_dy", "var_dy",
             "mix_entropy", "entropy", "kurt_dx", 'kurt_dy',
             "excess_entropy"]
fig_titles = ["Mean zonal displacement (in deg. longitude)",
              "Mean meridional displacement (in deg. latitude)",
              "Variance of zonal displacement (in deg. longitude sq.)",
              "Covariance of zonal and meridional displacement"
              + " (in deg. longitude deg. latitude)",
              "Variance of meridional displacement (in deg. latitude sq.)",
              "Mixture entropy", "Information entropy",
              "Excess kurtosis of zonal displacement",
              "Excess kurtosis of meridional displacement",
              "Excess information entropy"]
cmaps = [cmocean.cm.delta, cmocean.cm.amp, cmocean.cm.matter,
         cmocean.cm.balance]

for i in range(5):
    if i == 0:
        pc_data = mean[..., 0].numpy()
        cmap = cmaps[0]
        LIM = max((-pc_data.min(), pc_data.max()))
        LIM = 1.5
        CLIM = [-LIM, LIM]
        EXTEND = 'both'
    elif i == 1:
        pc_data = mean[..., 1].numpy()
        cmap = cmaps[0]
        LIM = max((-pc_data.min(), pc_data.max()))
        CLIM = [-LIM, LIM]
        EXTEND = 'both'
    elif i == 2:
        pc_data = cov[..., 0, 0].numpy()
        cmap = cmaps[1]
        CLIM = [0., 1.]
        EXTEND = 'max'
    elif i == 3:
        pc_data = cov[..., 0, 1].numpy()
        cmap = cmaps[0]
        LIM = max((-pc_data.min(), pc_data.max()))
        CLIM = [-LIM, LIM]
        EXTEND = 'both'
    elif i == 4:
        pc_data = cov[..., 1, 1].numpy()
        cmap = cmaps[1]
        CLIM = [0., 0.7]
        EXTEND = 'max'

    plt.figure(figsize=(6, 3))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
    ax.spines['geo'].set_visible(False)
    # ax.gridlines(draw_labels=True, dms=True,
    #              x_inline=False, y_inline=False)
    plt.title(fig_titles[i])
    sca = ax.pcolormesh(grid.vertices[..., 0], grid.vertices[..., 1],
                        pc_data,
                        cmap=cmap,
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
