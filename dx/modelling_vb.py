"""
Training script for Gaussian mixture density model of single-particle
transition density as a function of initial position.
"""

import sys
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import callbacks as cb
from tensorflow_probability.python.bijectors import fill_scale_tril
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.preprocessing import Scaler  # noqa: E402


tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfa = tf.keras.activations
tfb = tfp.bijectors
kl = tfd.kullback_leibler
tf.keras.backend.set_floatx("float64")

# Model hyperparameters
N_C = 1
DT = 4
MIXTURE_DENSITY = 'gaussian_mixture'

MODEL_DIR = f"dx/models/GDP_{DT:.0f}day_NC{N_C}_vb_flipout_Adam_tanh_lr5em5/"

if not Path(MODEL_DIR).exists():
    Path(MODEL_DIR).mkdir(parents=True)


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

X_ = Xscaler.standardise(X)
Y_ = Yscaler.standardise(Y)
del X, Y

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
        bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
        kernel_divergence_fn=(
            lambda q, p, ignore: kl.kl_divergence(q, p) / X_.shape[0]),
        bias_divergence_fn=(
            lambda q, p, ignore: kl.kl_divergence(q, p) / X_.shape[0]),
        activation=activation)


activation_fn = 'tanh'

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
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

# --- TRAIN MODEL ---

LOG_FILE = "log.csv"
CHECKPOINT_FILE = ("checkpoint_epoch_{epoch:02d}/weights")
TRAINED_FILE = "trained/weights"

# Training configuration


def nll(data_point, tf_distribution):
    """ Negative log likelihood. """
    return -tf_distribution.log_prob(data_point)


LOSS = nll
BATCH_SIZE = 8192
LEARNING_RATE = 5e-5
EPOCHS = 1000
OPTIMISER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
VALIDATION_SPLIT = 0

# Callbacks
CSV_LOGGER = cb.CSVLogger(MODEL_DIR + LOG_FILE)
BATCHES_PER_EPOCH = int(np.ceil(X_.shape[0] / BATCH_SIZE
                                * (1 - VALIDATION_SPLIT)))
CHECKPOINTING = cb.ModelCheckpoint(MODEL_DIR + CHECKPOINT_FILE,
                                   save_freq=10 * BATCHES_PER_EPOCH,
                                   verbose=1,
                                   save_weights_only=True)
EARLY_STOPPING = cb.EarlyStopping(monitor='loss', patience=20, min_delta=0.)
CALLBACKS = [CHECKPOINTING, CSV_LOGGER, EARLY_STOPPING]

# Model compilation and training
model.compile(loss=LOSS, optimizer=OPTIMISER)

History = model.fit(X_, Y_,
                    epochs=EPOCHS,
                    callbacks=CALLBACKS,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_split=VALIDATION_SPLIT,
                    verbose=2)

model.save_weights(MODEL_DIR + TRAINED_FILE)
