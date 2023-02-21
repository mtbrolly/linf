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

X = np.load(MODEL_DIR + "homogeneous_release_10year_rejection_v2.npy")

# m = grids.GTGP(n_x=180, n_y=90)  # !!!
# legit = (X[:, -1, 1] < 90.) * (X[:, -1, 1] > -90.)
# legit *= (X[:, -1, 0] < 180.) * (X[:, -1, 0] > -180.)
# X0 = X[legit, -1, :]
# m.count_X0s(X0, X0)


# Plotting

# pc_data = m.count.squeeze()

T_years = [0, 1, 3, 10]
T = [int(i * 365 / 4) for i in T_years]

fig, axs = plt.subplots(
    nrows=2, ncols=2,
    subplot_kw={'projection': cartopy.crs.Robinson(central_longitude=0.)},
    # subplot_kw={'projection': cartopy.crs.PlateCarree(central_longitude=0.)},
    figsize=(6, 4.2))

axs = axs.flatten()

for i in range(len(T)):
    m = grids.GTGP(n_x=180, n_y=90)
    X_T = X[:, T[i]]
    # legit = (X_T[:, 1] < 90.) * (X_T[:, 1] > -90.)
    # X_T = X_T[legit]
    m.count_X0s(X_T, X_T)
    pc_data = m.count.squeeze()

    sca = axs[i].pcolormesh(m.vertices[..., 0], m.vertices[..., 1], pc_data.T,
                            cmap=cmocean.cm.amp,
                            norm=colors.BoundaryNorm(
                                boundaries=[0, 1, 2, 5, 10, 15],
                                ncolors=256, extend='max'),
                            shading='flat',
                            transform=cartopy.crs.PlateCarree())

    # sca = axs[i].contourf(m.centres[..., 0], m.centres[..., 1], pc_data.T,
    #                       levels=[0, 1, 5, 10, 20, 100, 250],
    #                       norm=colors.BoundaryNorm(
    #                           boundaries=[0, 1, 5, 10, 20, 100, 250],
    #                           ncolors=256, clip=True),
    #                       cmap=cmocean.cm.tempo,
    #                       transform=cartopy.crs.PlateCarree())

    axs[i].add_feature(cartopy.feature.NaturalEarthFeature(
        "physical", "land", "50m"),
                    facecolor='k', edgecolor=None, zorder=100)
    axs[i].set_title(rf"$t=$ {T_years[i]:.0f} years")
# plt.tight_layout()
fig.subplots_adjust(bottom=0.14, top=0.95, left=0.02, right=0.98,
                    wspace=0.06, hspace=0.06
                    )

cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.035])  # [x0, y0, width, height]
cbar = fig.colorbar(sca, cax=cbar_ax, orientation='horizontal', extend='max')


plt.savefig(
    MODEL_DIR
    + "figures/homogeneous_release_10year_rejection/hists.png")
plt.close()
