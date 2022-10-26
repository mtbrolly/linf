"""
Script for analysis of, and figures relating to, dx models.
"""

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

# Model hyperparameters
N_C = 1
DT = 4

MODEL_DIR = f"dx/models/GDP_{DT:.0f}day_NC{N_C}_vb/"

CHECKPOINT = "trained"

print("Configuration done.")

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

print("Data prepared.")


# --- BUILD MODEL ---

# Data attributes
O_SIZE = len(Yscaler.mean)


DENSITY_PARAMS_SIZE = tfpl.MixtureSameFamily.params_size(
    N_C, component_params_size=tfpl.MultivariateNormalTriL.params_size(O_SIZE))


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


mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = tf.keras.Sequential([
        var_layer(256, 'relu'),
        var_layer(256, 'relu'),
        var_layer(256, 'relu'),
        var_layer(256, 'relu'),
        var_layer(512, 'relu'),
        var_layer(512, 'relu'),
        var_layer(N_C * 6, None),
        tfpl.MixtureSameFamily(N_C, tfpl.MultivariateNormalTriL(2))]
    )


# Load weights
model.load_weights(MODEL_DIR + CHECKPOINT + "/weights")

print("Model loaded.")

# Plot summary statistic on cartopy plot.
RES = 3.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)

gms_ = grid.eval_on_grid(model, scaler=Xscaler.standardise)

print("gms_ calculated.")

means = []
covs = []
for i in range(100):
    print(i)
    gms_ = grid.eval_on_grid(model, scaler=Xscaler.standardise)
    means.append(
        Yscaler.invert_standardisation_loc(gms_.mean())[None, ...])
    covs.append(
        Yscaler.invert_standardisation_cov(gms_.covariance())[None, ...])

means = tf.concat(means, axis=0)
covs = tf.concat(covs, axis=0)

mean_of_mean = tf.math.reduce_mean(means, axis=0)
mean_of_cov = tf.math.reduce_mean(covs, axis=0)
std_of_mean = tf.math.reduce_std(means, axis=0)
std_of_cov = tf.math.reduce_std(covs, axis=0)

np.save(MODEL_DIR + "mean_of_mean.npy", mean_of_mean)
np.save(MODEL_DIR + "mean_of_cov.npy", mean_of_cov)
np.save(MODEL_DIR + "std_of_mean.npy", std_of_mean)
np.save(MODEL_DIR + "std_of_cov.npy", std_of_cov)
