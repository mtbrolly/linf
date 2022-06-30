import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm
import cartopy
import cartopy.crs as ccrs
import cmocean
import pickle
import grids

# model_dir = "models/eddie2004/"
model_dir = "models/eddie0205_bs8192_lr5em4/"


def plot_conditional_density(model_dir):
    tf.keras.backend.set_floatx('float64')
    plt.style.use('./figures/experiments.mplstyle')
    # plt.ioff()

    figures_dir = model_dir + "figures_ES/"
    # trained_model_file = model_dir + "trained_nn"
    trained_model_file = model_dir + "checkpoint_nn_20_4475.83"

    # Load neural network and Gaussian mixture layer.
    with open(model_dir + 'gm.pickle', 'rb') as f:
        gm = pickle.load(f)
    NN = tf.keras.models.load_model(trained_model_file,
                                    custom_objects={'nll_reg': gm.nll_reg})
    gm.neural_net = NN

    scalers = []
    scalers_str = ["Xscaler", "Yscaler"]
    for i in range(len(scalers_str)):
        with open(model_dir + scalers_str[i] + '.pickle', 'rb') as f:
            scalers.append(pickle.load(f))

    [Xscaler, Yscaler] = scalers

    locations = {'gulf_stream': [-77.08, 32.71], 'irish_sea': [-5.2, 53.0],
                 'gulf_of_mexico': [-90., 25.], 'gulf_of_guinea': [0., 0.]}

    # location = np.array(locations['gulf_stream'])
    location = np.array([-130.82, 46.27])
    res = 100  # Grid points per degree
    degree_range = 5.
    plot_range = 5.
    xlims = (location[0] - degree_range, location[0] + degree_range)
    ylims = (location[1] - degree_range, location[1] + degree_range)
    grid = grids.LonlatGrid(n_x=2 * degree_range * res,
                            n_y=2 * degree_range * res,
                            xlims=xlims, ylims=ylims)
    grid_centres = np.concatenate((grid.centres[..., 0:1],
                                   grid.centres[..., 1:2]), axis=2)
    gmx = gm.get_gms_from_x(Xscaler.standardise(location[None, :]))

    alphas_all = gmx.mixture_distribution.probs.numpy().flatten()
    sig_comp = alphas_all > 0.05
    alphas = alphas_all[sig_comp]
    alpha_entropy = -np.sum(alphas_all * np.log2(alphas_all))
    means = Yscaler.invert_standardisation_loc(
        gmx.components_distribution.loc.numpy()[0, sig_comp, :])
    scale_trils = gmx.components_distribution.scale_tril.numpy()[
        0, sig_comp, :, :]
    covs = np.zeros_like(scale_trils)
    for i in range(sum(sig_comp)):
        covs[i, ...] = scale_trils[i, ...] @ scale_trils[i, ...].T
    covs = Yscaler.invert_standardisation_cov(covs)

    prob = Yscaler.invert_standardisation_prob(
        gmx.prob(Yscaler.standardise(grid_centres - location))).numpy()
    log_prob = np.log(prob)

    plt.figure(figsize=(16, 9))
    # plt.xkcd()
    map_proj = ccrs.PlateCarree(central_longitude=0.)
    # map_proj._threshold /= 10000.
    ax = plt.axes(projection=map_proj)
    # ax = plt.axes(projection=ccrs.Orthographic(central_latitude=-90.))
    ax.set_extent([location[0] - plot_range, location[0] + plot_range,
                   location[1] - plot_range, location[1] + plot_range],
                  crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True, dms=True,
                 x_inline=False, y_inline=False)
    plt.title("PDF of 2-day displacements from a point")
    sca = ax.pcolormesh(grid.vertices[..., 0], grid.vertices[..., 1], log_prob,
                        cmap=cmocean.cm.amp,
                        clim=[-2, None],
                        shading='flat',
                        transform=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='k',
                   facecolor=cartopy.feature.COLORS['land_alt1'])
    ax.coastlines()
    plt.colorbar(sca)
    ax.scatter(location[0], location[1], c='g', s=200., marker='x',
               transform=ccrs.PlateCarree())
    for i in range(sum(sig_comp)):
        ax.scatter(location[0] + means[i, 0],
                   location[1] + means[i, 1], s=200., marker='.',
                   color=cm.Greys(alphas[i]),
                   transform=ccrs.PlateCarree())
        ell = Ellipse((location[0] + means[i, 0], location[1] + means[i, 1]),
                      covs[i, 0, 0] ** 0.5, covs[i, 1, 1] ** 0.5,
                      color=cm.Greys(alphas[i]),
                      transform=ccrs.Geodetic(), fill=False)
        ax.add_patch(ell)
    plt.tight_layout()
    plt.show()
