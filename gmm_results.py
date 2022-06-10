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
plt.style.use('./figures/experiments.mplstyle')

model_dir = "models/ggm_1degree/"
figures_dir = model_dir + "figures/"


# Load model.
with open(model_dir + "ggm.pickle", 'rb') as f:
    m = pickle.load(f)


# Load data and scalers.
data_dir = "models/no_land/"
datas = []
datas_str = ["X", "XVAL", "Y", "YVAL", "X_", "X_VAL", "Y_", "Y_VAL"]
for i in range(len(datas_str)):
    datas.append(np.load(data_dir + datas_str[i] + '.npy'))
[X, XVAL, Y, YVAL, X_, X_VAL, Y_, Y_VAL] = datas


# Plotting
# =============================================================================
# pc_data = m.count[..., 0]
# pc_data = m.cov[..., 0, 0]
# pc_data[m.count.squeeze() < 10, ...] += np.nan
# lim = 1.
#
# plt.figure(figsize=(16, 9))
# ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
# # ax = plt.axes(projection=ccrs.Orthographic(central_latitude=-90.))
# ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# # plt.title(r"Drifter days per squared degree")
# bounds = np.array([0., 1., 10., 25., 50., 75., 100., 250.])
# norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
# sca = ax.pcolormesh(m.vertices[..., 0], m.vertices[..., 1],
#                     pc_data.T,
#                     # cmap=cmocean.cm.tempo,
#                     # norm=norm,
#                     cmap=cmocean.cm.amp,
#                     clim=[0., lim],
#                     shading='flat',
#                     transform=ccrs.PlateCarree())
# # sca = ax.contourf(m.centres[..., 0], m.centres[..., 1], pc_data,
# #                   cmap=cmocean.cm.tempo,
# #                   # levels=10,
# #                   norm=norm,
# #                   transform=ccrs.PlateCarree())
# # ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='k',
# #                facecolor=cartopy.feature.COLORS['land_alt1'])
# ax.coastlines()
# plt.colorbar(sca)
# plt.tight_layout()
# plt.show()
# # plt.savefig("models/ggm_1degree/figures/drifter_hours.pdf", format='pdf')
#
# =============================================================================

# Plot summary statistic on cartopy plot.
fig_names = ["mean_dx", "mean_dy", "var_dx", "cov_dx_dy", "var_dy"]
fig_titles = ["Mean zonal displacement", "Mean meridional displacement",
              "Variance of zonal displacement",
              "Covariance of zonal and meridional displacement",
              "Variance of meridional displacement"]
cmaps = [cmocean.cm.delta, cmocean.cm.amp]

for i in range(5):
    if i == 0:
        pc_data = m.mean[..., 0]
        cmap = cmaps[0]
        lim = max((-pc_data.min(), pc_data.max()))
        clim = [-lim, lim]
    elif i == 1:
        pc_data = m.mean[..., 1]
        cmap = cmaps[0]
        lim = max((-pc_data.min(), pc_data.max()))
        clim = [-lim, lim]
    elif i == 2:
        pc_data = m.cov[..., 0, 0]
        cmap = cmaps[1]
        clim = [0., pc_data.max()]
    elif i == 3:
        pc_data = m.cov[..., 0, 1]
        cmap = cmaps[0]
        lim = max((-pc_data.min(), pc_data.max()))
        clim = [-lim, lim]
    elif i == 4:
        pc_data = m.cov[..., 1, 1]
        cmap = cmaps[1]
        clim = [0., pc_data.max()]

    pc_data[m.count.squeeze() < 10, ...] += np.nan

    plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
    # ax = plt.axes(projection=ccrs.Orthographic(central_latitude=-90.))
    ax.gridlines(draw_labels=True, dms=True,
                 x_inline=False, y_inline=False)
    plt.title(fig_titles[i])
    # sca = ax.contourf(grid.centres[0], grid.centres[1], pc_data,
    #                   cmap=cmocean.cm.amp,
    #                   levels=np.linspace(clim[0], clim[1], 25),
    #                   transform=ccrs.PlateCarree())
    sca = ax.pcolormesh(m.vertices[..., 0], m.vertices[..., 1],
                        pc_data.T,
                        cmap=cmap,
                        clim=clim,
                        shading='flat',
                        transform=ccrs.PlateCarree())
    # ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='k',
    #                 facecolor=cartopy.feature.COLORS['land_alt1'])
    ax.coastlines()
    plt.colorbar(sca)
    plt.tight_layout()
    plt.savefig(figures_dir + fig_names[i] + ".pdf")
    plt.close()
