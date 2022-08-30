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
# import cartopy.crs as ccrs
from tensorflow.keras import callbacks as cb
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.preprocessing import Scaler  # noqa: E402

tfkl = tf.keras.layers
tfpl = tfp.layers
tf.keras.backend.set_floatx("float64")

DT = 2

assert DT in (2, 4), "Data not prepared for this value of DT."

if DT == 2:
    MODEL_DIR = "dx/models/GDP_2day_ml_periodic/"
else:
    MODEL_DIR = "dx/models/GDP_4day_ml_periodic/"

if not Path(MODEL_DIR).exists():
    Path(MODEL_DIR).mkdir(parents=True)


# --- PREPARE DATA ---

if DT == 2:
    DATA_DIR = "data/GDP/2day/"
else:
    DATA_DIR = "data/GDP/4day/"

X = np.load(DATA_DIR + "X0_train.npy")
Y = np.load(DATA_DIR + "DX_train.npy")

Xws = X.copy()
Xws[:, 0] -= 360.
Xes = X.copy()
Xes[:, 0] += 360.

# Periodicising X0.
X = np.concatenate((X, Xes, Xws), axis=0)
Y = np.concatenate((Y, Y, Y), axis=0)

# =============================================================================
# # Stereographic projection of X0.
# NPS = ccrs.NorthPolarStereo()
# X = NPS.transform_points(ccrs.PlateCarree(), X[:, 0], X[:, 1])[:, :2]
# =============================================================================

Xscaler = Scaler(X)
Yscaler = Scaler(Y)

X_ = Xscaler.standardise(X)
Y_ = Yscaler.standardise(Y)
del X, Y

# --- BUILD MODEL ---

# Data attributes
O_SIZE = Y_.shape[-1]

# Model hyperparameters
N_C = 32

DENSITY_PARAMS_SIZE = tfpl.MixtureSameFamily.params_size(
    N_C, component_params_size=tfpl.MultivariateNormalTriL.params_size(O_SIZE))

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = tf.keras.Sequential([
        tfkl.Dense(256, activation='relu'),
        tfkl.Dense(256, activation='relu'),
        tfkl.Dense(256, activation='relu'),
        tfkl.Dense(256, activation='relu'),
        tfkl.Dense(512, activation='relu'),
        tfkl.Dense(512, activation='relu'),
        tfkl.Dense(DENSITY_PARAMS_SIZE),
        tfpl.MixtureSameFamily(N_C, tfpl.MultivariateNormalTriL(O_SIZE))]
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
LEARNING_RATE = 5e-4
EPOCHS = 100
OPTIMISER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
VALIDATION_SPLIT = 0.1

# Callbacks
CSV_LOGGER = cb.CSVLogger(MODEL_DIR + LOG_FILE)
BATCHES_PER_EPOCH = int(np.ceil(X_.shape[0] / BATCH_SIZE
                                * (1 - VALIDATION_SPLIT)))
CHECKPOINTING = cb.ModelCheckpoint(MODEL_DIR + CHECKPOINT_FILE,
                                   save_freq=1 * BATCHES_PER_EPOCH,
                                   verbose=1,
                                   save_weights_only=True)
EARLY_STOPPING = cb.EarlyStopping(monitor='val_loss', patience=10)
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
