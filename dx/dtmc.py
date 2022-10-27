"""
DTMC model.
"""
import numpy as np
import pickle
from pathlib import Path
from tools import grids
import matplotlib.pyplot as plt
import cartopy
ccrs = cartopy.crs


# Create model
grid_res = 30.  # Grid resolution in degrees.
m = grids.DTMC(n_x=int(360. / grid_res), n_y=int(180. / grid_res))


# Load data

DT = 2

DATA_DIR = f"data/GDP/{DT:.0f}day/"

X0 = np.load(DATA_DIR + "X0_train.npy")
DX = np.load(DATA_DIR + "DX_train.npy")

m.fit(X0, DX)

X0v = np.load(DATA_DIR + "X0_test.npy")
DXv = np.load(DATA_DIR + "DX_test.npy")

mean_log_likelihood = m.mean_log_likelihood(X0v, DXv)
print(mean_log_likelihood)

###

X1 = X0 + DX

X1[(X1 > 180)[:, 0], 0] -= 360.
X1[(X1 < -180)[:, 0], 0] += 360.

# Get bin indices for X0 and X1
X0_lon_bin = np.digitize(X0[:, 0], m.vertices[0, :, 0])
X1_lon_bin = np.digitize(X1[:, 0], m.vertices[0, :, 0])
X0_lat_bin = np.digitize(X0[:, 1], m.vertices[:, 0, 1])
X1_lat_bin = np.digitize(X1[:, 1], m.vertices[:, 0, 1])

X0_bin = (X0_lat_bin - 1) * m.centres.shape[1] + (X0_lon_bin - 1)
X1_bin = (X1_lat_bin - 1) * m.centres.shape[1] + (X1_lon_bin - 1)

X0_none = np.bincount(X0_bin, minlength=m.transition_matrix.shape[0]) == 0

assert X0_bin.max() < m.transition_matrix.shape[0], "bin error"


plt.figure()
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
ax.set_global()
ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor=None,
               facecolor='k',
               linewidth=0.3)
sca = ax.pcolormesh(m.vertices[..., 0], m.vertices[..., 1],
                    m.transition_matrix_4d[60, 75, ...],
                    transform=ccrs.PlateCarree(),
                    cmap='Reds', shading='flat')
plt.colorbar(sca)
plt.show()

# Train model.
# m.fit(X, Y)

# Save model.
MODEL_DIR = "dx/models/dtmc/{grid_res:.0f}deg/"
if not Path(MODEL_DIR).exists():
    Path(MODEL_DIR).mkdir(parents=True)
with open(MODEL_DIR + "dtmc.pickle", 'wb') as f:
    pickle.dump(m, f, pickle.HIGHEST_PROTOCOL)
