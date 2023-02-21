"""
Script for plotting precomputed statistics from MDN model of transition
density.
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

MODEL_DIR = (f"dx/models/GDP_{DT:.0f}day_NC{N_C}/")

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
         label=r'Training loss')
plt.plot(range(1, len(history) + 1), history['val_loss'], color='grey',
         linestyle='--',
         label=r'Test loss')
plt.xlim(0, None)
plt.xlabel(r'$\mathrm{Epoch}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR + "loss.png")
plt.close()


# Plot summary statistic on cartopy plot.
RES = 3.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)

mean = np.load(MODEL_DIR + "mean.npy")
cov = np.load(MODEL_DIR + "cov.npy")
skew = np.load(MODEL_DIR + "skew.npy")
kurt = np.load(MODEL_DIR + "kurt.npy")
mix_ent = np.load(MODEL_DIR + "mix_ent.npy")


fig_names = ["mean_dx", "mean_dy",
             "var_dx", "cov_dx_dy", "var_dy",
             "skew_dx", "skew_dy",
             "kurt_dx", "kurt_dy",
             "mix_ent",
             "diff_minor", "diff_major"]
cmaps = [cmocean.cm.delta, cmocean.cm.amp, cmocean.cm.matter,
         cmocean.cm.balance]
lon_deg_to_m = grid.R * np.deg2rad(1) * np.cos(np.deg2rad(
    grid.centres[..., 1]))
lat_deg_to_m = grid.R * np.deg2rad(1)
sec_per_day = 60 * 60 * 24


for i in range(len(fig_names)):

    if i == 0:  # Mean dx
        pc_data = mean[..., 0].copy()
        pc_data *= lon_deg_to_m
        cmap = cmaps[0]
        pc_data /= 1000.
        # LIM = DT * 100000.
        LIM = 150.  # np.max(np.abs(pc_data))
        NORM = None
        CLIM = [-LIM, LIM]
        EXTEND = 'both'

    elif i == 1:  # Mean dy
        pc_data = mean[..., 1].copy()
        pc_data *= lat_deg_to_m
        pc_data /= 1000.
        cmap = cmaps[0]
        LIM = 150.
        CLIM = [-LIM, LIM]
        EXTEND = 'both'

    elif i == 2:  # Var dx
        pc_data = cov[..., 0, 0].copy()
        pc_data *= lon_deg_to_m ** 2
        pc_data /= 1000. ** 2
        cmap = cmaps[1]
        CLIM = [0, 25000]
        NORM = None  # colors.LogNorm(vmin=100)
        EXTEND = 'max'

    elif i == 3:  # Cov dxdy
        pc_data = cov[..., 0, 1].copy()
        pc_data *= lon_deg_to_m * lat_deg_to_m
        pc_data /= 1000. ** 2
        cmap = cmocean.cm.curl
        CLIM = None
        NORM = colors.CenteredNorm(halfrange=10000)
        EXTEND = None

    elif i == 4:  # Var dy
        pc_data = cov[..., 1, 1].copy()
        pc_data *= lat_deg_to_m ** 2
        pc_data /= 1000. ** 2
        cmap = cmaps[1]
        CLIM = [0, 12500]
        NORM = None
        EXTEND = 'max'

    elif i == 5:  # Skew dx
        pc_data = skew[..., 0].copy()
        cmap = cmocean.cm.tarn
        CLIM = [-2.5, 2.5]
        NORM = None
        EXTEND = None

    elif i == 6:  # Skew dy
        pc_data = skew[..., 1].copy()
        cmap = cmocean.cm.tarn
        CLIM = [-2.5, 2.5]
        NORM = None
        EXTEND = None

    elif i == 7:  # Kurt dx
        pc_data = kurt[..., 0].copy()
        cmap = cmocean.cm.balance
        CLIM = None  # [-10, 10]
        NORM = colors.TwoSlopeNorm(0., vmin=-3., vmax=10.)
        EXTEND = None

    elif i == 8:  # Kurt dy
        pc_data = kurt[..., 1].copy()
        cmap = cmocean.cm.balance
        CLIM = None  # [-10, 10]
        NORM = colors.TwoSlopeNorm(0., vmin=-3., vmax=10.)
        EXTEND = None

    elif i == 9:  # Mixture entropy
        pc_data = mix_ent.copy()
        cmap = cmocean.cm.dense
        CLIM = [0., np.log(32)]
        NORM = None
        EXTEND = None

    elif i == 10:  # Minor principal component of diffusivity
        cov_matrices = cov.copy()
        cov_matrices[..., 0, 0] *= lon_deg_to_m ** 2
        cov_matrices[..., 0, 1] *= lon_deg_to_m * lat_deg_to_m
        cov_matrices[..., 1, 0] *= lon_deg_to_m * lat_deg_to_m
        cov_matrices[..., 1, 1] *= lat_deg_to_m ** 2
        diff_matrices = cov_matrices / (2 * DT * sec_per_day)
        vals, vecs = np.linalg.eig(diff_matrices.copy())
        minor_vec = np.where(
            np.tile(vals[..., 0:1], (1, 1, 2))
            < np.tile(vals[..., 1:2], (1, 1, 2)),
            vecs[..., 0], vecs[..., 1])
        minor_val = np.where(vals[..., 0: 1] < vals[..., 1: 2],
                             vals[..., 0: 1], vals[..., 1: 2])
        pc_data = minor_val.squeeze()
        cmap = cmaps[1]
        CLIM = [0, 25000]
        NORM = None
        EXTEND = 'max'

    elif i == 11:  # Major principal component of diffusivity
        cov_matrices = cov.copy()
        cov_matrices[..., 0, 0] *= lon_deg_to_m ** 2
        cov_matrices[..., 0, 1] *= lon_deg_to_m * lat_deg_to_m
        cov_matrices[..., 1, 0] *= lon_deg_to_m * lat_deg_to_m
        cov_matrices[..., 1, 1] *= lat_deg_to_m ** 2
        diff_matrices = cov_matrices / (2 * DT * sec_per_day)
        vals, vecs = np.linalg.eig(diff_matrices.copy())
        minor_vec = np.where(
            np.tile(vals[..., 0:1], (1, 1, 2))
            < np.tile(vals[..., 1:2], (1, 1, 2)),
            vecs[..., 0], vecs[..., 1])
        minor_val = np.where(vals[..., 0: 1] > vals[..., 1: 2],
                             vals[..., 0: 1], vals[..., 1: 2])
        pc_data = minor_val.squeeze()
        cmap = cmaps[1]
        CLIM = [0, 25000]
        NORM = None
        EXTEND = 'max'

    plt.figure(figsize=(6, 3))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
    ax.spines['geo'].set_visible(False)

    sca = ax.pcolormesh(grid.vertices[..., 0], grid.vertices[..., 1],
                        pc_data,
                        cmap=cmap,
                        norm=NORM,
                        clim=CLIM,
                        shading='flat',
                        transform=ccrs.PlateCarree())

    ax.add_feature(cartopy.feature.NaturalEarthFeature(
        "physical", "land", "50m"),
                   facecolor='k', edgecolor=None, zorder=100)
    plt.colorbar(sca, extend=EXTEND, shrink=0.8,
                 )
    plt.tight_layout()

    plt.savefig(FIG_DIR + fig_names[i] + ".png")
    plt.close()
