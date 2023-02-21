import cmocean
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import cartopy
from tools import grids
from dx.utils import load_mdn, load_scalers

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

MODEL_DIR = (f"dx/models/GDP_{DT:.0f}day_NC{N_C}/")

model = load_mdn(DT=DT, N_C=N_C)
Xscaler, Yscaler = load_scalers(DT=DT, N_C=N_C)


# Plot summary statistic on cartopy plot.
RES = 2.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)

# X0 = np.array([-72.55, 33.67])[None, :]
X0 = np.array([-74.5, 34.85])[None, :]
lims = [-90, -55, 20, 50]

gm_ = model(Xscaler.standardise(X0))


def p_X1_given_X0(X1):
    """
    Evaluates transition density for fixed X_0.
    """
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

ax.add_feature(cartopy.feature.NaturalEarthFeature(
    "physical", "land", "50m"),
    facecolor='k', edgecolor=None, zorder=100)
plt.colorbar(sca, extend='min')
plt.tight_layout()
plt.savefig(MODEL_DIR + "figures/cond_gulf_stream2.png")
plt.close()
