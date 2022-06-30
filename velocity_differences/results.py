import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# model_dir = "velocity_differences/models/testVD/"


def results(model_dir):
    tf.keras.backend.set_floatx('float64')
    plt.style.use('./figures/experiments.mplstyle')
    plt.ioff()

    figures_dir = model_dir + "figures/"
    trained_model_file = model_dir + "trained_nn"
    # trained_model_file = model_dir + "checkpoint_epoch_10"

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

    r_train = np.load("/home/s1511699/github/linf/data/velocity_differences/"
                      + "r_train.npy")  # Fix path.
    r = np.linspace(r_train.min(), r_train.max(), 1000).reshape((-1, 1))
    gms_ = gm.get_gms_from_x(Xscaler.standardise(r))
    mean = Yscaler.invert_standardisation_loc(gms_.mean()).numpy()
    cov = Yscaler.invert_standardisation_cov(gms_.covariance()).numpy()

    plt.figure()
    plt.loglog(r, mean[:, 0], 'k')
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\langle \delta u_L \rangle$')
    plt.savefig(figures_dir + "mean_dul.pdf")
    plt.close()

    plt.figure()
    plt.loglog(r, cov[:, 0, 0], 'k')
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\langle \delta u_L \rangle$')
    plt.savefig(figures_dir + "var_dul.pdf")
    plt.close()
