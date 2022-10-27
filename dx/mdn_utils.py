import sys
import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.preprocessing import Scaler  # noqa: E402
from tools import grids  # noqa: E402

tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
kl = tfd.kullback_leibler
tf.keras.backend.set_floatx("float64")


def mdn_mean_log_likelihood(X0val, DXval, MODEL_DIR, DT, N_C,
                            o_size=2, checkpoint="trained"):

    # MODEL_DIR = f"dx/models/GDP_{DT:.0f}day_NC{N_C}_vb/"

    # --- PREPARE DATA ---

    DATA_DIR = f"data/GDP/{DT:.0f}day/"

    X = np.load(DATA_DIR + "X0_train.npy")
    Y = np.load(DATA_DIR + "DX_train.npy")

    Xws = X.copy()
    Xws[:, 0] -= 360.
    Xes = X.copy()
    Xes[:, 0] += 360.

    # Periodicising X0.
    X = np.concatenate((X, Xes, Xws), axis=0)
    Y = np.concatenate((Y, Y, Y), axis=0)

    Xscaler = Scaler(X)
    Yscaler = Scaler(Y)
    X_size = X.shape[0]

    del X, Y

    # --- BUILD MODEL ---

    def dense_layer(N, activation):
        return tfkl.Dense(N, activation=activation)

    def var_layer(N, activation):
        return tfpl.DenseFlipout(
            N,
            kernel_divergence_fn=(
                lambda q, p, ignore: kl.kl_divergence(q, p) / X_size),
            bias_divergence_fn=(
                lambda q, p, ignore: kl.kl_divergence(q, p) / X_size),
            activation=activation)

    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    model = tf.keras.Sequential([
        var_layer(256, 'relu'),
        var_layer(256, 'relu'),
        var_layer(256, 'relu'),
        var_layer(256, 'relu'),
        var_layer(512, 'relu'),
        var_layer(512, 'relu'),
        var_layer(N_C * 6, None),
        tfpl.MixtureSameFamily(N_C, tfpl.MultivariateNormalTriL(2))]  # !!! 32
    )

    # Load weights
    model.load_weights(MODEL_DIR + checkpoint + "/weights")

    gm_ = model(Xscaler.standardise(X0val))
    mean_log_likelihood = np.log(
        Yscaler.invert_standardisation_prob(
            np.exp(gm_.log_prob(Yscaler.standardise(DXval))))).mean()

    return mean_log_likelihood
