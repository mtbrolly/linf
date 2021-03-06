import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from du.tfp_build_model import build_model

model_dir = "du/models/du_iso_1907/"
checkpoint_file = "checkpoint_epoch_46"


def results(model_dir, checkpoint_file):
    tf.keras.backend.set_floatx('float64')
    plt.style.use('./misc/experiments.mplstyle')
    plt.ioff()

    figures_dir = model_dir + "figures/"

    history_file = "log.csv"

    # Load neural network and Gaussian mixture layer.
    MDN = build_model(model_dir)
    MDN.load_weights(model_dir + checkpoint_file + "/weights")

    scalers = []
    scalers_str = ["Xscaler", "Yscaler"]
    for i in range(len(scalers_str)):
        with open(model_dir + scalers_str[i] + '.pickle', 'rb') as f:
            scalers.append(pickle.load(f))

    [Xscaler, Yscaler] = scalers

    # Load history.
    history = pd.read_csv(model_dir + history_file)
    N_EPOCHS = history.shape[0]

    # History plots
    plt.figure()
    plt.plot(range(1, N_EPOCHS + 1), history['loss'], 'k',
             label='Training loss')
    plt.plot(range(1, N_EPOCHS + 1), history['val_loss'], 'grey',
             label='Test loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figures_dir + "loss.png", dpi=576)
    plt.close()

    r_train = np.load(model_dir + "X.npy")
    # r = np.linspace(r_train.min(), r_train.max(), 1000).reshape((-1, 1))
    r = np.logspace(np.log10(r_train.min()), np.log10(r_train.max()),
                    1000).reshape((-1, 1))
    gms_ = MDN(Xscaler.standardise(r))
    mean = Yscaler.invert_standardisation_loc(gms_.mean()).numpy()
    cov = Yscaler.invert_standardisation_cov(gms_.covariance()).numpy()

    # def s3(x): return gm.S3(x, sample_size=20000, block_size=100)
    # S3l, S3t = s3(Xscaler.standardise(r))
    # S3l = S3l * Yscaler.std[0] ** 3
    # S3t = S3t * Yscaler.std[0] * Yscaler.std[1] ** 2

    # def skewness(x): return gm.skewness(x, sample_size=10000, block_size=200)

    # skewnesses = skewness(Xscaler.standardise(r))

    # def kurtosis(x): return gm.kurtosis(x, sample_size=20000, block_size=100)

    # kurtoses = kurtosis(Xscaler.standardise(r))

    plt.figure()
    plt.plot(r, mean[:, 0], 'k', label=r'$\langle \delta u_L \rangle$')
    plt.plot(r, mean[:, 1], 'k--', label=r'$\langle \delta u_T \rangle$')
    plt.xlabel(r'$r$')
    plt.legend()
    plt.tight_layout()
    # plt.savefig(figures_dir + "S1.png", dpi=576)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    # plt.savefig(figures_dir + "S1_loglog.png", dpi=576)
    # plt.close()

    plt.figure()
    plt.plot(r, cov[:, 0, 0] + cov[:, 1, 1], 'k', label=r'$S_2(r)$')
    plt.plot(r, cov[:, 0, 0], 'k--', label=r'$S_2^{(L)}(r)$')
    plt.plot(r, cov[:, 1, 1], 'k-.', label=r'$S_2^{(T)}(r)$')
    plt.xlabel(r'$r$')
    plt.vlines(2 * np.pi / 64, 4e-2, 0.5, 'grey', '-.',
               label=r'$l_f$')
    plt.vlines(2 * np.pi / 6, 4e-2, 0.5, 'grey', '--',
               label=r'$l_{lsf}$')
    plt.vlines(2 * np.pi / 350, 4e-2, 0.5, 'grey', ':',
               label=r'$l_{d}$')
    plt.plot([0.04, 0.08], 16 * np.array([0.04, 0.08]) ** 2., 'g--',
             label=r'$r^{2}$')
    plt.plot([0.11, 0.25], np.array([0.11, 0.25]) ** (1.), 'b--',
             label=r'$r^{1}$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(figures_dir + "S2.png", dpi=576)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    # plt.savefig(figures_dir + "S2_loglog.png", dpi=576)
    # plt.close()

    # plt.figure()
    # plt.plot(r, S3l + S3t, 'k', label=r'$S_3(r)$')
    # plt.xlabel(r'$r$')
    # plt.vlines(2 * np.pi / 64, 4e-2, 0.5, 'grey', '-.',
    #            label=r'$l_f$')
    # plt.vlines(2 * np.pi / 6, 4e-2, 0.5, 'grey', '--',
    #            label=r'$l_{lsf}$')
    # plt.vlines(2 * np.pi / 350, 4e-2, 0.5, 'grey', ':',
    #            label=r'$l_{d}$')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(figures_dir + "S3.png", dpi=576)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.tight_layout()
    # plt.savefig(figures_dir + "S3_loglog.png",
    #             dpi=576)
    # plt.close()

    # plt.figure()
    # plt.plot(r, np.abs(S3l + S3t), 'k', label=r'$|S_3(r)|$')
    # plt.xlabel(r'$r$')
    # plt.vlines(2 * np.pi / 64, 4e-2, 0.5, 'grey', '-.',
    #            label=r'$l_f$')
    # plt.vlines(2 * np.pi / 6, 4e-2, 0.5, 'grey', '--',
    #            label=r'$l_{lsf}$')
    # plt.vlines(2 * np.pi / 350, 4e-2, 0.5, 'grey', ':',
    #            label=r'$l_{d}$')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(figures_dir + "S3_abs.png", dpi=576)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.tight_layout()
    # plt.savefig(figures_dir + "S3_abs_loglog.png",
    #             dpi=576)
    # plt.close()

    # plt.figure()
    # plt.plot(r, np.abs(skewnesses[:, 0]),
    #          'k', label=r'$\mathrm{Skew}(\delta u_L | r)$')
    # plt.plot(r, np.abs(skewnesses[:, 1]), 'grey',
    #          label=r'$\mathrm{Skew}(\delta u_T | r)$')
    # plt.xlabel(r'$r$')
    # # plt.yscale('log')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(figures_dir + "skew.png", dpi=576)
    # plt.close()

    # plt.figure()
    # plt.plot(r, kurtoses[:, 0], 'k', label=r'$\mathrm{Kurt}(\delta u_L | r)$')
    # plt.plot(r, kurtoses[:, 1], 'grey',
    #          label=r'$\mathrm{Kurt}(\delta u_T | r)$')
    # plt.plot(r[:, 0], r[:, 0] * 0. + 3., 'g--', label=r'$\mathrm{Gaussian}$')
    # plt.xlabel(r'$r$')
    # plt.grid(True)
    # plt.legend()
    # # plt.tight_layout()
    # # plt.savefig(figures_dir + "kurt.png", dpi=576)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.tight_layout()
    # plt.savefig(figures_dir + "kurt_loglog.png",
    #             dpi=576)
    # plt.close()
