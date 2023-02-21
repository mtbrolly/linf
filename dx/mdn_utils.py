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
tfa = tf.keras.activations
kl = tfd.kullback_leibler
tf.keras.backend.set_floatx("float64")


def mdn_mean_log_likelihood(X0val, DXval, DT, N_C, block_size=20000):
    """
    Computes the mean log likelihood of data under the MDN model.
    """

    MODEL_DIR = (f"dx/models/GDP_{DT:.0f}day_NC{N_C}/")

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

    # Data attributes
    O_SIZE = len(Yscaler.mean)

    DENSITY_PARAMS_SIZE = int(tfpl.MixtureSameFamily.params_size(
        N_C, component_params_size=tfpl.MultivariateNormalTriL.params_size(
            O_SIZE)))
    mixture_density_layer = tfpl.MixtureSameFamily(
        N_C, tfpl.MultivariateNormalTriL(O_SIZE))

    def dense_layer(N, activation):
        return tfkl.Dense(N, activation=activation)

    def var_layer(N, activation):
        return tfpl.DenseFlipout(
            N,
            bias_posterior_fn=tfpl.util.default_mean_field_normal_fn(
                ),
            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn=(
                lambda q, p, ignore: kl.kl_divergence(q, p) / X_size),
            bias_divergence_fn=(
                lambda q, p, ignore: kl.kl_divergence(q, p) / X_size),
            activation=activation)

    activation_fn = 'tanh'

    model = tf.keras.Sequential([
        dense_layer(256, activation_fn),
        dense_layer(256, activation_fn),
        dense_layer(256, activation_fn),
        dense_layer(256, activation_fn),
        dense_layer(512, activation_fn),
        dense_layer(512, activation_fn),
        dense_layer(DENSITY_PARAMS_SIZE, None),
        mixture_density_layer]
    )

    # Load weights

    model.load_weights(MODEL_DIR + "trained/weights")

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
