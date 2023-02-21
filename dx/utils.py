"""
Utility functions for analysis of MDN models.
"""

import pickle
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfkl = tf.keras.layers
tfpl = tfp.layers
tf.keras.backend.set_floatx("float64")


def load_mdn(DT, N_C):
    """
    Loads MDN model.
    """

    model = tf.keras.Sequential(
        [tfkl.Dense(256, activation='tanh'),
         tfkl.Dense(256, activation='tanh'),
         tfkl.Dense(256, activation='tanh'),
         tfkl.Dense(256, activation='tanh'),
         tfkl.Dense(512, activation='tanh'),
         tfkl.Dense(512, activation='tanh'),
         tfkl.Dense(N_C * 6, activation=None),
         tfpl.MixtureSameFamily(N_C, tfpl.MultivariateNormalTriL(2))])

    model.load_weights(f"dx/models/GDP_{DT:.0f}day_NC{N_C}/trained/weights")
    return model


def load_scalers(DT, N_C):
    """
    Loads scaler objects relating to MDN models.
    """

    with open(f"dx/models/GDP_{DT:.0f}day_NC{N_C}/Xscaler.pkl", "rb") as file:
        Xscaler = pickle.load(file)

    with open(f"dx/models/GDP_{DT:.0f}day_NC{N_C}/Yscaler.pkl", "rb") as file:
        Yscaler = pickle.load(file)
    return Xscaler, Yscaler


def mdn_mean_log_likelihood(X0val, DXval, DT, N_C, block_size=20000):
    """
    Computes the mean log likelihood of data under the MDN model.
    """

    model = load_mdn(DT=DT, N_C=N_C)
    Xscaler, Yscaler = load_scalers(DT=DT, N_C=N_C)

    def mll(X0val, DXval):
        gm_ = model(Xscaler.standardise(X0val))
        mean_log_likelihood = np.log(
            Yscaler.invert_standardisation_prob(
                np.exp(gm_.log_prob(Yscaler.standardise(DXval))))).mean()
        return mean_log_likelihood

    mlls = []
    for i in range(int(np.ceil(X0val.shape[0] / block_size))):
        mlls.append(mll(X0val[i * block_size: (i + 1) * block_size, :],
                        DXval[i * block_size: (i + 1) * block_size, :]))
        print('mll of block calculated')
    mean_log_likelihood = np.mean(np.array(mlls))
    return mean_log_likelihood
