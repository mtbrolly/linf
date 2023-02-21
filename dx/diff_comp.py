import sys
import os
import cmocean
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import grids  # noqa: E402

ccrs = cartopy.crs
plt.style.use('./misc/paper.mplstyle')
plt.ioff()

comp = 'minor'
cov_4day = np.load("dx/models/GDP_4day_NC32/" + "cov.npy")

cov_14day = np.load("dx/models/GDP_14day_NC32/" + "cov.npy")


RES = 3.
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)

lon_deg_to_m = grid.R * np.deg2rad(1) * np.cos(np.deg2rad(
    grid.centres[..., 1]))
lat_deg_to_m = grid.R * np.deg2rad(1)
sec_per_day = 60 * 60 * 24

DT = 4
cov_matrices = cov_4day.copy()
cov_matrices[..., 0, 0] *= lon_deg_to_m ** 2
cov_matrices[..., 0, 1] *= lon_deg_to_m * lat_deg_to_m
cov_matrices[..., 1, 0] *= lon_deg_to_m * lat_deg_to_m
cov_matrices[..., 1, 1] *= lat_deg_to_m ** 2
diff_matrices = cov_matrices / (2 * DT * sec_per_day)
vals, vecs = np.linalg.eig(diff_matrices.copy())
minor_val = np.where(vals[..., 0: 1] < vals[..., 1: 2],
                     vals[..., 0: 1], vals[..., 1: 2])
major_val = np.where(vals[..., 0: 1] > vals[..., 1: 2],
                     vals[..., 0: 1], vals[..., 1: 2])
if comp == 'major':
    diff_4 = major_val.squeeze()
else:
    diff_4 = minor_val.squeeze()


DT = 14
cov_matrices = cov_14day.copy()
cov_matrices[..., 0, 0] *= lon_deg_to_m ** 2
cov_matrices[..., 0, 1] *= lon_deg_to_m * lat_deg_to_m
cov_matrices[..., 1, 0] *= lon_deg_to_m * lat_deg_to_m
cov_matrices[..., 1, 1] *= lat_deg_to_m ** 2
diff_matrices = cov_matrices / (2 * DT * sec_per_day)
vals, vecs = np.linalg.eig(diff_matrices.copy())
minor_val = np.where(vals[..., 0: 1] < vals[..., 1: 2],
                     vals[..., 0: 1], vals[..., 1: 2])
major_val = np.where(vals[..., 0: 1] > vals[..., 1: 2],
                     vals[..., 0: 1], vals[..., 1: 2])
if comp == 'major':
    diff_14 = major_val.squeeze()
else:
    diff_14 = minor_val.squeeze()


# pc_data = diff_14 / diff_4
# pc_data = (diff_14 - diff_4) / diff_4
# pc_data = np.log2(diff_14 / diff_4)
pc_data = diff_14 - diff_4

# pc_data[diff_4 < 5000] = np.nan

plt.figure(figsize=(6, 3))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
ax.spines['geo'].set_visible(False)
sca = ax.pcolormesh(grid.vertices[..., 0], grid.vertices[..., 1],
                    pc_data,
                    cmap='RdBu_r',
                    norm=colors.CenteredNorm(halfrange=25000.),
                    clim=[-25000., 25000.],
                    shading='flat',
                    transform=ccrs.PlateCarree())

# sca = ax.contourf(grid.centres[..., 0], grid.centres[..., 1],
#                   pc_data,
#                   cmap='RdBu_r',
#                   # levels=[-2, -1, -0.6, -0.3, 0.3, 0.6, 1, 2],
#                   levels=[0.1, 0.2, 0.5, 0.66, 1.5, 2., 5., 10.],
#                   norm=colors.BoundaryNorm(
#                       # [-2, -1, -0.6, -0.3, 0.3, 0.6, 1, 2],
#                       [0.1, 0.2, 0.5, 0.66, 1.5, 2., 5., 10.],
#                       mpl.cm.get_cmap('RdBu_r').N, clip=True),
#                   transform=ccrs.PlateCarree())

ax.add_feature(cartopy.feature.NaturalEarthFeature(
    "physical", "land", "50m"),
               facecolor='k', edgecolor=None, zorder=100)
plt.colorbar(sca, extend='both', shrink=0.8)
plt.tight_layout()
plt.savefig("dx/models/GDP_4day_NC32/"
            + "figures/diffs_difference_" + comp + ".png")

plt.close('all')
