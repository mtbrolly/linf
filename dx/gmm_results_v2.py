"""
Gridded Gaussian model.
"""
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cmocean
import pickle
plt.style.use('misc/paper.mplstyle')
plt.ioff()

model_dir = "dx/models/ggm/1deg/"
figures_dir = model_dir + "figures/"


# Load model.
with open(model_dir + "ggm.pickle", 'rb') as f:
    m = pickle.load(f)


# Load data

DT = 2

if DT == 2:
    DATA_DIR = "data/GDP/2day/"
else:
    DATA_DIR = "data/GDP/4day/"

X = np.load(DATA_DIR + "X0_train.npy")
Y = np.load(DATA_DIR + "DX_train.npy")

# Concatenate with test data for map of drifter coverage

XVAL = np.load(DATA_DIR + "X0_test.npy")
YVAL = np.load(DATA_DIR + "DX_test.npy")

X = np.concatenate((X, XVAL))
Y = np.concatenate((Y, YVAL))

del XVAL, YVAL


# Plotting
pc_data = m.count.squeeze()

plt.figure(figsize=(6, 3))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
ax.spines['geo'].set_visible(False)

bounds = np.array([0., 1., 25., 50., 100., 250., 1000., 2000., 4000., 7000.])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, clip=True)

sca = ax.pcolormesh(m.vertices[..., 0], m.vertices[..., 1], pc_data.T,
                    norm=norm,
                    cmap=cmocean.cm.tempo,
                    # cmap='jet',
                    shading='flat',
                    transform=ccrs.PlateCarree())

# sca = ax.contourf(m.centres[..., 0], m.centres[..., 1], pc_data.T,
#                   levels=bounds,
#                   norm=norm,
#                   # cmap=cmocean.cm.tempo,
#                   cmap='jet',
#                   transform=ccrs.PlateCarree())

ax.add_feature(cartopy.feature.LAND, zorder=100, facecolor='k')
plt.colorbar(sca)
plt.tight_layout()
# plt.show()

plt.savefig("dx/models/ggm/1deg/figures/drifter_coverage.png")
