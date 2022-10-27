"""
Script for analysis of, and figures relating to, dx models.
"""

import sys
import os
from pathlib import Path
import cmocean
# import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.preprocessing import Scaler  # noqa: E402
from tools import grids  # noqa: E402

ccrs = cartopy.crs
# tfkl = tf.keras.layers
# tfpl = tfp.layers
# tfd = tfp.distributions
# kl = tfd.kullback_leibler
# tf.keras.backend.set_floatx("float64")
plt.style.use('./misc/paper.mplstyle')
plt.ioff()

# Model hyperparameters
N_C = 1
DT = 4

MODEL_DIR = f"dx/models/GDP_{DT:.0f}day_NC{N_C}_vb/"

CHECKPOINT = "trained"
FIG_DIR = MODEL_DIR + "figures/"
if not Path(FIG_DIR).exists():
    Path(FIG_DIR).mkdir(parents=True)

print("Configuration done.")

# --- PREPARE DATA ---

DATA_DIR = f"data/GDP/{DT:.0f}day/"

# Load log
history = pd.read_csv(MODEL_DIR + "log.csv")


# --- LOSS PLOT ---

plt.figure()
plt.plot(range(1, len(history) + 1), history['loss'], 'k',
         label='Training loss')
plt.xlabel(r'$\mathrm{Epoch}$')
plt.ylabel(r'$\mathrm{Training loss}$')
# plt.legend()
plt.grid()
plt.tight_layout()
plt.ylim(history['loss'].min(), history['loss'][0])
plt.savefig(FIG_DIR + "loss.png")
plt.close()


# Plot summary statistic on cartopy plot.
RES = 3.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)

mean_of_mean = np.load(MODEL_DIR + "mean_of_mean.npy")
mean_of_cov = np.load(MODEL_DIR + "mean_of_cov.npy")
std_of_mean = np.load(MODEL_DIR + "std_of_mean.npy")
std_of_cov = np.load(MODEL_DIR + "std_of_cov.npy")


fig_names = ["mean_dx", "mean_u", "mean_dy", "mean_v",
             "var_dx", "diff_x", "cov_dx_dy", "diff_xy", "var_dy", "diff_y",
             "cv_diff_x", "cv_diff_y",
             "cv_mean_dx", "cv_mean_dy",
             "cv_var_dx", "cv_diff_x", "cv_cov_dx_dy"]
cmaps = [cmocean.cm.delta, cmocean.cm.amp, cmocean.cm.matter,
         cmocean.cm.balance]
lon_deg_to_m = grid.R * np.deg2rad(1) * np.cos(np.deg2rad(
    grid.centres[..., 1]))
lat_deg_to_m = grid.R * np.deg2rad(1)
sec_per_day = 24 * 60 * 60

mean = mean_of_mean
cov = mean_of_cov

for i in range(14):
    if i == 0:  # Mean dx
        pc_data = mean[..., 0].copy()
        pc_data *= lon_deg_to_m
        cmap = cmaps[0]
        LIM = 100000.
        NORM = None
        CLIM = [-LIM, LIM]
        EXTEND = 'both'
    elif i == 1:  # Mean u
        pc_data = mean[..., 0].copy()
        pc_data *= lon_deg_to_m / (2 * DT * sec_per_day)
        cmap = cmaps[0]
        LIM = 0.5  # np.max(np.abs(pc_data))
        CLIM = [-LIM, LIM]
        EXTEND = 'both'
    elif i == 2:  # Mean dy
        pc_data = mean[..., 1].copy()
        pc_data *= lat_deg_to_m
        cmap = cmaps[0]
        LIM = 100000.
        CLIM = [-LIM, LIM]
        EXTEND = 'both'
    elif i == 3:  # Mean v
        pc_data = mean[..., 1].copy()
        pc_data *= lon_deg_to_m / (2 * DT * sec_per_day)
        cmap = cmaps[0]
        LIM = 0.5  # np.max(np.abs(pc_data))
        CLIM = [-LIM, LIM]
        EXTEND = 'both'
    elif i == 4:  # Var dx
        pc_data = cov[..., 0, 0].copy()
        pc_data *= lon_deg_to_m ** 2
        cmap = cmaps[1]
        LIM = 1e10
        CLIM = [0., LIM]
        # CLIM = None
        # NORM = colors.LogNorm(1e7, 1e10)
        EXTEND = 'max'
    elif i == 5:  # Diff xx
        pc_data = cov[..., 0, 0].copy()
        pc_data *= lon_deg_to_m ** 2 / (2 * DT * sec_per_day)
        cmap = cmaps[1]
        NORM = colors.LogNorm(1e3, 3e4)
        CLIM = None
        EXTEND = 'both'
    elif i == 6:  # Cov dxdy
        pc_data = cov[..., 0, 1].copy()
        pc_data *= lon_deg_to_m * lat_deg_to_m
        cmap = cmaps[0]
        LIM = 5e9
        CLIM = [-LIM, LIM]
        NORM = None
        EXTEND = 'both'
    elif i == 7:  # Diff xy
        pc_data = cov[..., 0, 1].copy()
        pc_data *= lon_deg_to_m * lat_deg_to_m / (2 * DT * sec_per_day)
        cmap = cmaps[0]
        LIM = 1e4
        CLIM = [-LIM, LIM]
        NORM = None
        EXTEND = 'both'
    elif i == 8:  # Var dy
        pc_data = cov[..., 1, 1].copy()
        pc_data *= lat_deg_to_m ** 2
        cmap = cmaps[1]
        LIM = 1e10
        CLIM = [0., LIM]
        NORM = None
        EXTEND = 'max'
    elif i == 9:  # Diff yy
        pc_data = cov[..., 1, 1].copy()
        pc_data *= lat_deg_to_m ** 2 / (2 * DT * sec_per_day)
        cmap = cmaps[1]
        NORM = colors.LogNorm(1e3, 3e4)
        CLIM = None
        EXTEND = 'both'
    elif i == 10:  # CV diff xx
        pc_data_std = std_of_cov[..., 0, 0].copy()
        pc_data = cov[..., 0, 0]
        pc_data = pc_data_std / pc_data
        cmap = 'jet'  # cmaps[1]
        NORM = colors.LogNorm(1e-2, 1e0)
        # NORM = None
        # CLIM = [1e-2, 1e0]
        EXTEND = 'both'
    elif i == 11:  # CV diff yy
        pc_data_std = std_of_cov[..., 1, 1].copy()
        pc_data = cov[..., 1, 1]
        pc_data = pc_data_std / pc_data
        cmap = 'jet'  # cmaps[1]
        NORM = colors.LogNorm(1e-2, 1e0)
        # NORM = None
        # CLIM = [1e-2, 1e0]
        EXTEND = 'both'
    elif i == 12:  # CV mean_dx
        pc_data_std = std_of_mean[..., 0].copy()
        pc_data = mean[..., 0]
        pc_data = np.abs(pc_data_std / pc_data)
        cmap = 'jet'  # cmaps[1]
        NORM = colors.LogNorm(1e-2, 1e0)
        # NORM = None
        # CLIM = [1e-2, 1e0]
        EXTEND = 'both'
    elif i == 13:  # CV mean_dy
        pc_data_std = std_of_mean[..., 1].copy()
        pc_data = mean[..., 1]
        pc_data = np.abs(pc_data_std / pc_data) * (pc_data > 0.05)
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
