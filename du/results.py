import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# model_dir = "du/models/vd0107_lr5em7/"


def results(model_dir, checkpoint_file):
    tf.keras.backend.set_floatx('float64')
    plt.style.use('./misc/experiments.mplstyle')
    plt.ioff()

    figures_dir = model_dir + "figures/"
    # trained_model_file = model_dir + "checkpoint_epoch_10"

    if not checkpoint_file:
        trained_model_file = model_dir + "trained_nn"
        history_file = "history.csv"
        second_training_prefix = ""
    else:
        trained_model_file = model_dir + "trained_from_" + checkpoint_file
        history_file = "history_since_" + checkpoint_file + ".csv"
        second_training_prefix = "second_training_"

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
    plt.tight_layout()
    plt.savefig(figures_dir + second_training_prefix + "loss.png", dpi=576)
    plt.close()

    r_train = np.load(model_dir + "X.npy")
    r = np.linspace(r_train.min(), r_train.max(), 1000).reshape((-1, 1))
    gms_ = gm.get_gms_from_x(Xscaler.standardise(r))
    mean = Yscaler.invert_standardisation_loc(gms_.mean()).numpy()
    cov = Yscaler.invert_standardisation_cov(gms_.covariance()).numpy()

    plt.figure()
    plt.plot(r, mean[:, 0], 'k', label=r'$\langle \delta u_L \rangle$')
    plt.plot(r, mean[:, 1], 'k--', label=r'$\langle \delta u_T \rangle$')
    plt.xlabel(r'$r$')
    plt.legend()
    plt.savefig(figures_dir + second_training_prefix + "S1.png", dpi=576)
    plt.close()

    plt.figure()
    plt.plot(r, cov[:, 0, 0] + cov[:, 1, 1], 'k', label=r'$S_2(r)$')
    plt.xlabel(r'$r$')
    plt.vlines(2 * np.pi / 64, 4e-2, 0.5, 'grey', '-.',
               label=r'$l_f$')
    plt.vlines(2 * np.pi / 6, 4e-2, 0.5, 'grey', '--',
               label=r'$l_{lsf}$')
    plt.vlines(2 * np.pi / 350, 4e-2, 0.5, 'grey', ':',
               label=r'$l_{d}$')
    plt.plot([0.01, 0.08], 124 * np.array([0.01, 0.08]) ** 2., 'r--',
             label=r'$r^{2}$')
    plt.plot([0.11, 0.25], np.array([0.11, 0.25]) ** (1.), 'm--',
             label=r'$r^{1}$')
    plt.grid(True)
    plt.legend()
    plt.savefig(figures_dir + second_training_prefix + "S2.png", dpi=576)
    plt.close()
