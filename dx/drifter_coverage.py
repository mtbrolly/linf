import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy
import cmocean
from tools import grids

plt.ioff()
plt.style.use("./misc/paper.mplstyle")

# Load data.

DT = 4.
DATA_DIR = f"data/GDP/{DT:.0f}day/"

X = np.load(DATA_DIR + "X0_train.npy")
Y = np.load(DATA_DIR + "DX_train.npy")

XVAL = np.load(DATA_DIR + "X0_test.npy")
YVAL = np.load(DATA_DIR + "DX_test.npy")

X = np.concatenate((X, XVAL))
Y = np.concatenate((Y, YVAL))

del XVAL, YVAL


# Create GTGP model.

m = grids.GTGP(n_x=360, n_y=180)
m.count_X0s(X, Y)


# Plotting
pc_data = m.count.squeeze()

plt.figure(figsize=(6, 3))
ax = plt.axes(projection=cartopy.crs.Robinson(central_longitude=0.0))
ax.spines["geo"].set_visible(False)

bounds = np.array([0.0, 1.0, 25.0, 50.0, 100.0,
                   250.0, 1000.0, 2000.0, 4000.0, 7000.0])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, clip=True)

sca = ax.pcolormesh(
    m.vertices[..., 0],
    m.vertices[..., 1],
    pc_data.T,
    norm=norm,
    cmap=cmocean.cm.tempo,
    shading="flat",
    transform=cartopy.crs.PlateCarree())



ax.add_feature(
    cartopy.feature.NaturalEarthFeature("physical", "land", "50m"),
    facecolor="k",
    edgecolor=None,
    zorder=100)

plt.colorbar(sca)
plt.tight_layout()

plt.savefig("./figures/drifter_coverage.png")
plt.close()
