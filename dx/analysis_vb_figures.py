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
plt.style.use('./misc/paper.mplstyle')
plt.ioff()

# Model hyperparameters
N_C = 32
DT = 4

MODEL_DIR = f"dx/models/GDP_{DT:.0f}day_NC{N_C}_vb_flipout_Adam_relu_lr5em5_pat50_val20_det_final2/"

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
plt.plot(range(1, len(history) + 1), history['val_loss'], 'r',
         label='Test loss')
# plt.yscale('log')
plt.xlabel(r'$\mathrm{Epoch}$')
# plt.ylabel(r'$\mathrm{Training and test loss}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR + "loss.png")

# Plot summary statistic on cartopy plot.
RES = 3.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)

mean_of_mean = np.load(MODEL_DIR + "mean_of_mean.npy")
mean_of_cov = np.load(MODEL_DIR + "mean_of_cov.npy")
# mean_of_mix_ent = np.load(MODEL_DIR + "mean_of_mix_ent.npy")
# mean_of_kurt = np.load(MODEL_DIR + "mean_of_kurt.npy")
std_of_mean = np.load(MODEL_DIR + "std_of_mean.npy")
std_of_cov = np.load(MODEL_DIR + "std_of_cov.npy")


fig_names = ["mean_dx", "mean_dy",
             "var_dx", "cov_dx_dy", "var_dy",
             "cv_var_dx", "cv_var_dy",
             "cv_mean_dx", "cv_mean_dy",
             "kurt_dx", "kurt_dy", "mix_ent"]
cmaps = [cmocean.cm.delta, cmocean.cm.amp, cmocean.cm.matter,
         cmocean.cm.balance]
lon_deg_to_m = grid.R * np.deg2rad(1) * np.cos(np.deg2rad(
    grid.centres[..., 1]))
lat_deg_to_m = grid.R * np.deg2rad(1)
sec_per_day = 24 * 60 * 60

mean = mean_of_mean
cov = mean_of_cov

for i in range(len(fig_names)):

    if i == 0:  # Mean dx
        pc_data = mean[..., 0].copy()
        pc_data *= lon_deg_to_m
        pc_data /= 1000.
        cmap = cmaps[0]
        LIM = DT * 100.
        NORM = None
        CLIM = [-LIM, LIM]
        EXTEND = 'both'

    elif i == 1:  # Mean dy
        pc_data = mean[..., 1].copy()
        pc_data *= lat_deg_to_m
        pc_data /= 1000.
        cmap = cmaps[0]
        LIM = 100.
        CLIM = [-LIM, LIM]
        EXTEND = 'both'

    elif i == 2:  # Var dx
        pc_data = cov[..., 0, 0].copy()
        pc_data *= lon_deg_to_m ** 2
        pc_data /= 1000. ** 2
        cmap = cmaps[1]
        CLIM = [0, 25000]
        NORM = None
        EXTEND = 'max'

    elif i == 3:  # Cov dxdy
        pc_data = cov[..., 0, 1].copy()
        pc_data *= lon_deg_to_m * lat_deg_to_m
        pc_data /= 1000. ** 2
        cmap = cmaps[0]
        CLIM = None
        NORM = colors.CenteredNorm(halfrange=15000)
        EXTEND = None

    elif i == 4:  # Var dy
        pc_data = cov[..., 1, 1].copy()
        pc_data *= lat_deg_to_m ** 2
        pc_data /= 1000. ** 2
        cmap = cmaps[1]
        CLIM = [0, 12500]
        NORM = None
        EXTEND = 'max'

    elif i == 5:  # CV var dx
        pc_data_std = std_of_cov[..., 0, 0].copy()
        pc_data = cov[..., 0, 0].copy()
        pc_data = pc_data_std / pc_data
        cmap = 'jet'  # cmaps[1]
        NORM = colors.LogNorm(1e-2, 1e0)
        # NORM = None
        CLIM = None
        EXTEND = 'both'

    elif i == 6:  # CV var dy
        pc_data_std = std_of_cov[..., 1, 1].copy()
        pc_data = cov[..., 1, 1].copy()
        pc_data = pc_data_std / pc_data
        cmap = 'jet'  # cmaps[1]
        NORM = colors.LogNorm(1e-2, 1e-0)
        # NORM = None
        # CLIM = [1e-2, 1e0]
        EXTEND = 'both'

    elif i == 7:  # CV mean dx
        pc_data_std = std_of_mean[..., 0].copy()
        pc_data = mean[..., 0].copy()
        pc_data = np.abs(pc_data_std / pc_data)
        cmap = 'jet'  # cmaps[1]
        NORM = colors.LogNorm(1e-2, 1e0)
        # NORM = None
        # CLIM = [1e-2, 1e0]
        EXTEND = 'both'

    elif i == 8:  # CV mean dy
        pc_data_std = std_of_mean[..., 1].copy()
        pc_data = mean[..., 1].copy()
        pc_data = np.abs(pc_data_std / pc_data) * (pc_data > 0.05)
        cmap = 'jet'  # cmaps[1]
        NORM = colors.LogNorm(1e-2, 1e0)
        # NORM = None
        # CLIM = [1e-2, 1e0]
        EXTEND = 'both'

    # elif i == 9:  # Kurt dx
    #     pc_data = mean_of_kurt[..., 0].copy()
    #     cmap = cmaps[0]
    #     NORM = colors.CenteredNorm()
    #     EXTEND = None

    # elif i == 10:  # Kurt dy
    #     pc_data = mean_of_kurt[..., 1].copy()
    #     cmap = cmaps[0]
    #     NORM = colors.CenteredNorm()
    #     EXTEND = None

    # elif i == 11:  # Mix entropy
    #     pc_data = mean_of_mix_ent.copy()
    #     cmap = cmaps[1]
    #     NORM = None
    #     CLIM = [0., None]
    #     EXTEND = None

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
    # ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor=None,
    #                facecolor='k',
    #                linewidth=0.3)
    ax.add_feature(cartopy.feature.NaturalEarthFeature(
        "physical", "land", "50m"),
                   facecolor='k', edgecolor=None, zorder=100)
    plt.colorbar(sca, extend=EXTEND, shrink=0.8,
                 # pad=0.05, orientation='horizontal', fraction=0.05
                 )
    plt.tight_layout()

    # plt.show()
    plt.savefig(FIG_DIR + fig_names[i] + ".png")
    plt.close()

plt.figure()
plt.plot(range(1, len(history) + 1), history['loss'], 'k',
         label='Training loss')
plt.plot(range(1, len(history) + 1), history['val_loss'], 'r',
         label='Test loss')
# plt.yscale('log')
plt.xlabel(r'$\mathrm{Epoch}$')
# plt.ylabel(r'$\mathrm{Training and test loss}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR + "loss.png")
# plt.ylim(history['loss'].min(), history['loss'][0])
# plt.yscale('log')
# plt.tight_layout()
# plt.savefig(FIG_DIR + "loss2.png")
# plt.close()


# Plot summary statistic on cartopy plot.
RES = 3.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)

mean_of_mean = np.load(MODEL_DIR + "mean_of_mean.npy")
mean_of_cov = np.load(MODEL_DIR + "mean_of_cov.npy")
# mean_of_mix_ent = np.load(MODEL_DIR + "mean_of_mix_ent.npy")
# mean_of_kurt = np.load(MODEL_DIR + "mean_of_kurt.npy")
std_of_mean = np.load(MODEL_DIR + "std_of_mean.npy")
std_of_cov = np.load(MODEL_DIR + "std_of_cov.npy")


fig_names = ["mean_dx", "mean_dy",
             "var_dx", "cov_dx_dy", "var_dy",
             "cv_var_dx", "cv_var_dy",
             "cv_mean_dx", "cv_mean_dy",
             "kurt_dx", "kurt_dy", "mix_ent"]
cmaps = [cmocean.cm.delta, cmocean.cm.amp, cmocean.cm.matter,
         cmocean.cm.balance]
lon_deg_to_m = grid.R * np.deg2rad(1) * np.cos(np.deg2rad(
    grid.centres[..., 1]))
lat_deg_to_m = grid.R * np.deg2rad(1)
sec_per_day = 24 * 60 * 60

mean = mean_of_mean
cov = mean_of_cov

for i in range(len(fig_names)):

    if i == 0:  # Mean dx
        pc_data = mean[..., 0].copy()
        pc_data *= lon_deg_to_m
        pc_data /= 1000.
        cmap = cmaps[0]
        LIM = DT * 100.
        NORM = None
        CLIM = [-LIM, LIM]
        EXTEND = 'both'

    elif i == 1:  # Mean dy
        pc_data = mean[..., 1].copy()
        pc_data *= lat_deg_to_m
        pc_data /= 1000.
        cmap = cmaps[0]
        LIM = 100.
        CLIM = [-LIM, LIM]
        EXTEND = 'both'

    elif i == 2:  # Var dx
        pc_data = cov[..., 0, 0].copy()
        pc_data *= lon_deg_to_m ** 2
        pc_data /= 1000. ** 2
        cmap = cmaps[1]
        CLIM = [0, 25000]
        NORM = None
        EXTEND = 'max'

    elif i == 3:  # Cov dxdy
        pc_data = cov[..., 0, 1].copy()
        pc_data *= lon_deg_to_m * lat_deg_to_m
        pc_data /= 1000. ** 2
        cmap = cmaps[0]
        CLIM = None
        NORM = colors.CenteredNorm(halfrange=15000)
        EXTEND = None

    elif i == 4:  # Var dy
        pc_data = cov[..., 1, 1].copy()
        pc_data *= lat_deg_to_m ** 2
        pc_data /= 1000. ** 2
        cmap = cmaps[1]
        CLIM = [0, 12500]
        NORM = None
        EXTEND = 'max'

    elif i == 5:  # CV var dx
        pc_data_std = std_of_cov[..., 0, 0].copy()
        pc_data = cov[..., 0, 0].copy()
        pc_data = pc_data_std / pc_data
        cmap = 'jet'  # cmaps[1]
        NORM = colors.LogNorm(1e-2, 1e0)
        # NORM = None
        CLIM = None
        EXTEND = 'both'

    elif i == 6:  # CV var dy
        pc_data_std = std_of_cov[..., 1, 1].copy()
        pc_data = cov[..., 1, 1].copy()
        pc_data = pc_data_std / pc_data
        cmap = 'jet'  # cmaps[1]
        NORM = colors.LogNorm(1e-2, 1e-1)
        # NORM = None
        # CLIM = [1e-2, 1e0]
        EXTEND = 'both'

    elif i == 7:  # CV mean dx
        pc_data_std = std_of_mean[..., 0].copy()
        pc_data = mean[..., 0].copy()
        pc_data = np.abs(pc_data_std / pc_data)
        cmap = 'jet'  # cmaps[1]
        NORM = colors.LogNorm(1e-2, 1e0)
        # NORM = None
        # CLIM = [1e-2, 1e0]
        EXTEND = 'both'

    elif i == 8:  # CV mean dy
        pc_data_std = std_of_mean[..., 1].copy()
        pc_data = mean[..., 1].copy()
        pc_data = np.abs(pc_data_std / pc_data) * (pc_data > 0.05)
        cmap = 'jet'  # cmaps[1]
        NORM = colors.LogNorm(1e-2, 1e0)
        # NORM = None
        # CLIM = [1e-2, 1e0]
        EXTEND = 'both'

    # elif i == 9:  # Kurt dx
    #     pc_data = mean_of_kurt[..., 0].copy()
    #     cmap = cmaps[0]
    #     NORM = colors.CenteredNorm()
    #     EXTEND = None

    # elif i == 10:  # Kurt dy
    #     pc_data = mean_of_kurt[..., 1].copy()
    #     cmap = cmaps[0]
    #     NORM = colors.CenteredNorm()
    #     EXTEND = None

    # elif i == 11:  # Mix entropy
    #     pc_data = mean_of_mix_ent.copy()
    #     cmap = cmaps[1]
    #     NORM = None
    #     CLIM = [0., None]
    #     EXTEND = None

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
    # ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor=None,
    #                facecolor='k',
    #                linewidth=0.3)
    ax.add_feature(cartopy.feature.NaturalEarthFeature(
        "physical", "land", "50m"),
                   facecolor='k', edgecolor=None, zorder=100)
    plt.colorbar(sca, extend=EXTEND, shrink=0.8,
                 # pad=0.05, orientation='horizontal', fraction=0.05
                 )
    plt.tight_layout()

    # plt.show()
    plt.savefig(FIG_DIR + fig_names[i] + ".png")
    plt.close()
