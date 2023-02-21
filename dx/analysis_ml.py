"""
Script for computing statistics from MDN model of transition density.
"""

import sys
import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.stats import kurtosis, skew
from dx.utils import load_mdn, load_scalers
sys.path.insert(1, os.path.join(sys.path[0], '..'))
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

model = load_mdn(DT=DT, N_C=N_C)
Xscaler, Yscaler = load_scalers(DT=DT, N_C=N_C)

# Compute statistics at the vertices of a longitude-latitude grid.

RES = 3.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)

gms_ = grid.eval_on_grid(model, scaler=Xscaler.standardise)

print("gms_ calculated.")

mean = Yscaler.invert_standardisation_loc(gms_.mean())
cov = Yscaler.invert_standardisation_cov(gms_.covariance())


def excess_kurtosis(sample_size, block_size):
    """
    Estimates excess kurtosis by Monte Carlo.
    """
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
    """
    Estimates skewness by Monte Carlo.
    """
    return skew(gms_.sample(sample_size), axis=0)


skewness = skewness_(1000)


np.save(MODEL_DIR + "mean.npy", mean)
np.save(MODEL_DIR + "cov.npy", cov)
np.save(MODEL_DIR + "kurt.npy", kurt)
np.save(MODEL_DIR + "mix_ent.npy", mix_ent)
np.save(MODEL_DIR + "skew.npy", skewness)
