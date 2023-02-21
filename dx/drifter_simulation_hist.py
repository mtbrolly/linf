import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy
import cmocean
from tools import grids
plt.ioff()
plt.style.use('./misc/paper.mplstyle')

N_C = 32
DT = 4

MODEL_DIR = (f"dx/models/GDP_{DT:.0f}day_NC{N_C}"
             + "_ml_flipout_Adam_tanh_lr5em5_pat50_val20/")

X = np.load(MODEL_DIR + "homogeneous_release_10year_rejection.npy")

m = grids.GTGP(n_x=180, n_y=90)  # !!!
legit = (X[:, -1, 1] < 90.) * (X[:, -1, 1] > -90.)
# legit *= (X[:, -1, 0] < 180.) * (X[:, -1, 0] > -1800.)
X0 = X[legit, -1, :]
m.count_X0s(X0, X0)


# Plotting
pc_data = m.count.squeeze()

plt.figure(figsize=(6, 3))
ax = plt.axes(projection=cartopy.crs.Robinson(central_longitude=0.))
# ax.spines['geo'].set_visible(False)

# sca = ax.pcolormesh(m.vertices[..., 0], m.vertices[..., 1], pc_data.T,
#                     # norm=norm,
#                     cmap=cmocean.cm.amp,  # tempo,
#                     # cmap='jet',
#                     clim=[None, 10.],
#                     shading='flat',
#                     transform=cartopy.crs.PlateCarree())

sca = ax.contourf(m.centres[..., 0], m.centres[..., 1], pc_data.T,
                  levels=[0, 1, 5, 10, 20, 90],
                  norm=colors.BoundaryNorm(
                      boundaries=[0, 1, 5, 10, 20, 90],
                      ncolors=256, clip=True),
                  cmap=cmocean.cm.tempo,
                  transform=cartopy.crs.PlateCarree())

ax.add_feature(cartopy.feature.NaturalEarthFeature(
    "physical", "land", "50m"),
               facecolor='k', edgecolor=None, zorder=100)
plt.colorbar(sca, extend='max')
plt.tight_layout()

# plt.show()
plt.savefig(
    MODEL_DIR
    + "figures/homogeneous_release_10year_rejection/final_hist2.png")
plt.close()
