import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy
import cartopy.crs as ccrs
import cmocean
import pickle
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import grids  # noqa: E402


def results(model_dir):
    tf.keras.backend.set_floatx('float64')
    plt.style.use('./figures/experiments.mplstyle')
    plt.ioff()

    figures_dir = model_dir + "figures/"
    trained_model_file = model_dir + "trained_nn"
    # trained_model_file = model_dir + "checkpoint_nn_20_4475.83"

    # Load neural network and Gaussian mixture layer.
    with open(model_dir + 'gm.pickle', 'rb') as f:
        gm = pickle.load(f)
    NN = tf.keras.models.load_model(trained_model_file,
                                    custom_objects={'nll_reg': gm.nll_reg})
    gm.neural_net = NN

    # Load data.
    # =============================================================================
    # datas = []
    # datas_str = ["X_", "X_VAL", "Y_", "Y_VAL"]
    # for i in range(len(datas_str)):
    #     datas.append(np.load(model_dir + datas_str[i] + '.npy'))
    #
    # [X_, X_VAL, Y_, Y_VAL] = datas
    # =============================================================================

    # Data attributes
    # =============================================================================
    # BATCH_SIZE = NN.layers[0].input_shape[0][0]
    # I_SIZE = NN.layers[0].input_shape[0][-1]
    # O_SIZE = gm.o_size
    # =============================================================================

    scalers = []
    scalers_str = ["Xscaler", "Yscaler"]
    for i in range(len(scalers_str)):
        with open(model_dir + scalers_str[i] + '.pickle', 'rb') as f:
            scalers.append(pickle.load(f))

    [Xscaler, Yscaler] = scalers

    # Load history.
    history = pd.read_csv(model_dir + "log.csv")
    N_EPOCHS = history.shape[0]

    # History plots
    plt.figure()
    plt.plot(range(1, N_EPOCHS + 1), history['loss'], 'k',
             label='Training loss')
    plt.plot(range(1, N_EPOCHS + 1), history['val_loss'], 'grey',
             label='Test loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir + "loss.pdf")
    plt.close()

    # Plot summary statistic on cartopy plot.
    res = 1.  # Grid points per degree
    grid = grids.LonlatGrid(n_x=360 * res, n_y=180 * res)
    gms_ = grid.eval_on_grid(gm.get_gms_from_x, scaler=Xscaler.standardise)
    mean = Yscaler.invert_standardisation_loc(gms_.mean())
    # cov_scaled = gms_.covariance()
    cov = Yscaler.invert_standardisation_cov(gms_.covariance())
    # mixture_entropies = grid.eval_on_grid(gm.mixture_entropy,
    #                                       scaler=Xscaler.standardise)

    # def entropy(x): return gm.entropy(x, sample_size=1000, block_size=1000)

    # entropies = grid.eval_on_grid(entropy, scaler=Xscaler.standardise)
    # Gaussian_entropy = 0.5 * np.log((2 * np.pi * np.exp(1)) ** 2 *
    #                                 (cov_scaled[:, 0, 0] * cov_scaled[:, 1, 1]
    #                                  - cov_scaled[:, 0, 1] ** 2))
    # excess_entropies = entropies - Gaussian_entropy

    # def kurtosis(x): return gm.kurtosis(x, sample_size=1000, block_size=1000)

    # kurtoses = grid.eval_on_grid(kurtosis, scaler=Xscaler.standardise)

    fig_names = ["mean_dx", "mean_dy", "var_dx", "cov_dx_dy", "var_dy",
                 "mix_entropy", "entropy", "kurt_dx", 'kurt_dy',
                 "excess_entropy"]
    fig_titles = ["Mean zonal displacement", "Mean meridional displacement",
                  "Variance of zonal displacement",
                  "Covariance of zonal and meridional displacement",
                  "Variance of meridional displacement",
                  "Mixture entropy", "Information entropy",
                  "Excess kurtosis of zonal displacement",
                  "Excess kurtosis of meridional displacement",
                  "Excess information entropy"]
    cmaps = [cmocean.cm.delta, cmocean.cm.amp, cmocean.cm.matter,
             cmocean.cm.balance]

    for i in range(5):
        if i == 0:
            pc_data = mean.numpy().reshape(
                grid.centres.shape[:-1] + (2,))[..., 0]
            cmap = cmaps[0]
            lim = max((-pc_data.min(), pc_data.max()))
            lim = 1.5
            clim = [-lim, lim]
            norm = None
        elif i == 1:
            pc_data = mean.numpy().reshape(
                grid.centres.shape[:-1] + (2,))[..., 1]
            cmap = cmaps[0]
            lim = max((-pc_data.min(), pc_data.max()))
            clim = [-lim, lim]
            norm = None
        elif i == 2:
            pc_data = cov.numpy().reshape(grid.centres.shape[:-1]
                                          + (2, 2))[..., 0, 0]
            cmap = cmaps[1]
            clim = [0., 1.]
            norm = None
        elif i == 3:
            pc_data = cov.numpy().reshape(grid.centres.shape[:-1]
                                          + (2, 2))[..., 0, 1]
            cmap = cmaps[0]
            lim = max((-pc_data.min(), pc_data.max()))
            clim = [-lim, lim]
            norm = None
        elif i == 4:
            pc_data = cov.numpy().reshape(grid.centres.shape[:-1]
                                          + (2, 2))[..., 1, 1]
            cmap = cmaps[1]
            clim = [0., pc_data.max()]
            norm = None
        # elif i == 5:
        #     pc_data = mixture_entropies.numpy().reshape(
        #         grid.centres.shape[:-1])
        #     cmap = cmaps[1]
        #     clim = [0., np.log(gm.n_c)]
        #     norm = None
        # elif i == 6:
        #     pc_data = entropies.reshape(grid.centres.shape[:-1])
        #     cmap = cmaps[2]
        #     clim = [0., 6.]
        #     norm = None
        # elif i == 7:
        #     pc_data = kurtoses[..., 0].reshape(grid.centres.shape[:-1]) - 3
        #     cmap = cmaps[3]
        #     clim = None
        #     norm = colors.SymLogNorm(linthresh=0.3, vmin=-10., vmax=10.)
        # elif i == 8:
        #     pc_data = kurtoses[..., 1].reshape(grid.centres.shape[:-1]) - 3
        #     cmap = cmaps[3]
        #     clim = None
        #     norm = colors.SymLogNorm(linthresh=0.3, vmin=-10., vmax=10.)
        # elif i == 9:
        #     pc_data = excess_entropies.reshape(grid.centres.shape[:-1])
        #     cmap = cmaps[2]
        #     clim = [-1., None]
        #     norm = None

        plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.Robinson())
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
        # ax.coastlines()
        plt.colorbar(sca)
        plt.tight_layout()

        # plt.savefig(figures_dir + fig_names[i] + ".pdf")
        plt.savefig(figures_dir + fig_names[i] + ".png")
        plt.close()
