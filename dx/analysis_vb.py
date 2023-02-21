"""
Script for analysis of, and figures relating to, dx models.
"""

import sys
import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow_probability.python.bijectors import fill_scale_tril
from scipy.stats import kurtosis
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.preprocessing import Scaler  # noqa: E402
from tools import grids  # noqa: E402


tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfa = tf.keras.activations
kl = tfd.kullback_leibler
tf.keras.backend.set_floatx("float64")

# Model hyperparameters
N_C = 1
DT = 4
MIXTURE_DENSITY = 'gaussian_mixture'

MODEL_DIR = f"dx/models/GDP_{DT:.0f}day_NC{N_C}_vb_flipout_Adam_tanh_lr5em5_long/"

CHECKPOINT = "checkpoint_epoch_5180"  # "trained"

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


if MIXTURE_DENSITY == 'gaussian_mixture':
    DENSITY_PARAMS_SIZE = int(tfpl.MixtureSameFamily.params_size(
        N_C, component_params_size=tfpl.MultivariateNormalTriL.params_size(
            O_SIZE)))
    mixture_density_layer = tfpl.MixtureSameFamily(
        N_C, tfpl.MultivariateNormalTriL(O_SIZE))

elif MIXTURE_DENSITY == 'single_student_t':
    DENSITY_PARAMS_SIZE = 1 + O_SIZE + O_SIZE * (O_SIZE + 1) // 2

    mixture_density_layer = tfpl.DistributionLambda(
        lambda t: tfd.MultivariateStudentTLinearOperator(
            df=4. + tfa.exponential(t[..., 0]),
            loc=t[..., 1:1 + O_SIZE],
            scale=tf.linalg.LinearOperatorLowerTriangular(
                fill_scale_tril.FillScaleTriL(
                    diag_shift=np.array(1e-5, t.dtype.as_numpy_dtype),
                    validate_args=True)(t[..., 1 + O_SIZE:]))
            ))


def dense_layer(N, activation):
    return tfkl.Dense(N, activation=activation)


def var_layer(N, activation):
    return tfpl.DenseFlipout(
        N,
        bias_posterior_fn=tfpl.util.default_mean_field_normal_fn(
            ),
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_divergence_fn=(
            lambda q, p, ignore: kl.kl_divergence(q, p) / X_size),
        bias_divergence_fn=(
            lambda q, p, ignore: kl.kl_divergence(q, p) / X_size),
        activation=activation)


activation_fn = 'tanh'

model = tf.keras.Sequential([
    var_layer(256, activation_fn),
    var_layer(256, activation_fn),
    var_layer(256, activation_fn),
    var_layer(256, activation_fn),
    var_layer(512, activation_fn),
    var_layer(512, activation_fn),
    var_layer(DENSITY_PARAMS_SIZE, None),
    mixture_density_layer]
)


# Load weights
model.load_weights(MODEL_DIR + CHECKPOINT + "/weights")

print("Model loaded.")

# Plot summary statistic on cartopy plot.
RES = 3.  # Grid points per degree
grid = grids.LonlatGrid(n_x=360 * RES, n_y=180 * RES)

# gms_ = grid.eval_on_grid(model, scaler=Xscaler.standardise)

print("gms_ calculated.")

means = []
covs = []
mix_ents = []
kurts = []
for i in range(100):
    print(i)
    gms_ = grid.eval_on_grid(model, scaler=Xscaler.standardise)
    means.append(
        Yscaler.invert_standardisation_loc(gms_.mean())[None, ...])
    covs.append(
        Yscaler.invert_standardisation_cov(gms_.covariance())[None, ...])
    mix_probs = tf.keras.activations.softmax(gms_.mixture_distribution.logits)
    mix_ents.append(
        -tf.reduce_sum(
            tf.math.multiply(
                mix_probs, tf.math.log(mix_probs)), axis=-1)[None, ...])

    def excess_kurtosis(sample_size, block_size):
        n_blocks = sample_size // block_size
        # sum_4th_powers = 0.
        # sum_squares = 0.

        for b in range(n_blocks):
            if b == 0:
                samples = gms_.sample(sample_size)  # !!!
            else:
                samples = tf.concat(
                    (samples, gms_.sample(sample_size)),axis=0)
            # sum_4th_powers += (samples - means[-1]) ** 4
            # sum_squares += (samples - means[-1]) ** 2

        return kurtosis(samples, axis=0)

    kurts.append(excess_kurtosis(1000, 50)[None, ...])

means = tf.concat(means, axis=0)
covs = tf.concat(covs, axis=0)
mix_ents = tf.concat(mix_ents, axis=0)
kurts = tf.concat(kurts, axis=0)

mean_of_mean = tf.math.reduce_mean(means, axis=0)
mean_of_cov = tf.math.reduce_mean(covs, axis=0)
mean_of_mix_ent = tf.math.reduce_mean(mix_ents, axis=0)
mean_of_kurt = tf.math.reduce_mean(kurts, axis=0)
std_of_mean = tf.math.reduce_std(means, axis=0)
std_of_cov = tf.math.reduce_std(covs, axis=0)
std_of_mix_ent = tf.math.reduce_std(mix_ents, axis=0)
std_of_kurt = tf.math.reduce_std(kurts, axis=0)

np.save(MODEL_DIR + "mean_of_mean.npy", mean_of_mean)
np.save(MODEL_DIR + "mean_of_cov.npy", mean_of_cov)
np.save(MODEL_DIR + "mean_of_mix_ent.npy", mean_of_mix_ent)
np.save(MODEL_DIR + "mean_of_kurt.npy", mean_of_kurt)
np.save(MODEL_DIR + "std_of_mean.npy", std_of_mean)
np.save(MODEL_DIR + "std_of_cov.npy", std_of_cov)
np.save(MODEL_DIR + "std_of_mix_ent.npy", std_of_mix_ent)
np.save(MODEL_DIR + "std_of_kurt.npy", std_of_kurt)
