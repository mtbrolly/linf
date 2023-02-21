"""
Script for computing statistics from MDN model of transition density.
"""

import sys
import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.stats import kurtosis, skew
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.preprocessing import Scaler  # noqa: E402
from tools import grids  # noqa: E402

tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
kl = tfd.kullback_leibler
tf.keras.backend.set_floatx("float64")

# Model hyperparameters
N_C = 32
DT = 4

MODEL_DIR = (f"dx/models/GDP_{DT:.0f}day_NC{N_C}/")

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

del X, Y, Xws, Xes

print("Data prepared.")


# --- BUILD MODEL ---

# Data attributes
O_SIZE = len(Yscaler.mean)


DENSITY_PARAMS_SIZE = tfpl.MixtureSameFamily.params_size(
    N_C, component_params_size=tfpl.MultivariateNormalTriL.params_size(O_SIZE))


def dense_layer(N, activation):
    return tfkl.Dense(N, activation=activation)


activation_fn = 'tanh'

model = tf.keras.Sequential([
    dense_layer(256, activation_fn),
    dense_layer(256, activation_fn),
    dense_layer(256, activation_fn),
    dense_layer(256, activation_fn),
    dense_layer(512, activation_fn),
    dense_layer(512, activation_fn),
    dense_layer(N_C * 6, None),
    tfpl.MixtureSameFamily(N_C, tfpl.MultivariateNormalTriL(2))]
)


# Load weights
model.load_weights(MODEL_DIR + CHECKPOINT + "/weights")

# Compute statistics at the vertices of a longitude-latitude grid.

RES = 3.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)

gms_ = grid.eval_on_grid(model, scaler=Xscaler.standardise)

print("gms_ calculated.")

mean = Yscaler.invert_standardisation_loc(gms_.mean())
cov = Yscaler.invert_standardisation_cov(gms_.covariance())


def excess_kurtosis(sample_size, block_size):
    n_blocks = sample_size // block_size
    for b in range(n_blocks):
        if b == 0:
            samples = gms_.sample(sample_size)
        else:
            samples = tf.concat(
                (samples, gms_.sample(sample_size)), axis=0)
    return kurtosis(samples, axis=0)


kurt = excess_kurtosis(1000, 200)


mix_probs = tf.keras.activations.softmax(gms_.mixture_distribution.logits)
mix_ent = -tf.reduce_sum(
    tf.math.multiply(mix_probs, tf.math.log(mix_probs)), axis=-1)


def skewness_(sample_size):
    return skew(gms_.sample(sample_size), axis=0)


skewness = skewness_(1000)


np.save(MODEL_DIR + "mean.npy", mean)
np.save(MODEL_DIR + "cov.npy", cov)
np.save(MODEL_DIR + "kurt.npy", kurt)
np.save(MODEL_DIR + "mix_ent.npy", mix_ent)
np.save(MODEL_DIR + "skew.npy", skewness)
