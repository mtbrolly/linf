import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import seaborn as sns
import cmocean
import sys
import os
from pathlib import Path
# from scipy.stats import norm
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.preprocessing import Scaler  # noqa: E402
import tools.grids as grids  # noqa: E402
tfd = tfp.distributions
tfa = tf.keras.activations
tf.keras.backend.set_floatx("float64")
plt.style.use('./misc/experiments.mplstyle')
cp = sns.color_palette("husl", 8)
plt.ioff()

tfkl = tf.keras.layers
tfpl = tfp.layers


MODEL_DIR = "dx/models/GDP/"
CHECKPOINT = "trained"
# CHECKPOINT = "checkpoint_epoch_01"
FIG_DIR = MODEL_DIR + "figures/"
if not Path(FIG_DIR).exists():
    Path(FIG_DIR).mkdir(parents=True)


# --- PREPARE DATA ---

DATA_DIR = "data/dx/"

X = np.load(DATA_DIR + "X0_train.npy")
Y = np.load(DATA_DIR + "DX_train.npy")

Xscaler = Scaler(X)
Yscaler = Scaler(Y)


# --- BUILD MODEL ---

# Data attributes
O_SIZE = Y.shape[-1]

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
plt.savefig(FIG_DIR + "loss.png", dpi=576)
plt.close()


# Plot summary statistic on cartopy plot.
res = 2.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * res, n_y=180 * res)

gms_ = grid.eval_on_grid(model, scaler=Xscaler.standardise)
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
        lim = max((-pc_data.min(), pc_data.max()))
        lim = 1.5
        clim = [-lim, lim]
        norm = None
    elif i == 1:
        pc_data = mean[..., 0].numpy()
        cmap = cmaps[0]
        lim = max((-pc_data.min(), pc_data.max()))
        clim = [-lim, lim]
        norm = None
    elif i == 2:
        pc_data = cov[..., 0, 0].numpy()
        cmap = cmaps[1]
        clim = [0., 1.]
        norm = None
    elif i == 3:
        pc_data = cov[..., 0, 1].numpy()
        cmap = cmaps[0]
        lim = max((-pc_data.min(), pc_data.max()))
        clim = [-lim, lim]
        norm = None
    elif i == 4:
        pc_data = cov[..., 1, 1].numpy()
        cmap = cmaps[1]
        clim = [0., 0.7]
        norm = None

    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
    ax.gridlines(draw_labels=True, dms=True,
                 x_inline=False, y_inline=False)
    plt.title(fig_titles[i])
    sca = ax.pcolormesh(grid.vertices[..., 0], grid.vertices[..., 1],
                        pc_data,
                        cmap=cmap,
                        clim=clim,
                        norm=norm,
                        shading='flat',
                        transform=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='k',
                   facecolor=cartopy.feature.COLORS['land_alt1'],
                   linewidth=0.3)
    plt.colorbar(sca)
    plt.tight_layout()

    # plt.savefig(FIG_DIR + fig_names[i] + ".png", dpi=576)
    plt.close()
