"""
Training script for Gaussian mixture density model of spatial velocity
increments conditioned on separation distance.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from pathlib import Path
import sys
import os
from tensorflow.keras import callbacks as cb
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.preprocessing import Scaler  # noqa: E402
tf.keras.backend.set_floatx("float64")

tfkl = tf.keras.layers
tfpl = tfp.layers


MODEL_DIR = "du/models/GLAD_1608/"

if not Path(MODEL_DIR).exists():
    Path(MODEL_DIR).mkdir(parents=True)


# --- PREPARE DATA ---

# DATA_DIR = "data/du/"

# X = np.load(DATA_DIR + "r_train.npy")
# Y = np.load(DATA_DIR + "du_train.npy")

DATA_DIR = "data/GLAD/"

X = np.load(DATA_DIR + "r.npy")[:, None]
Y = np.load(DATA_DIR + "du.npy")

Xscaler = Scaler(X)
Yscaler = Scaler(Y)

X_ = Xscaler.standardise(X)
Y_ = Yscaler.standardise(Y)


# --- BUILD MODEL ---

# Data attributes
O_SIZE = Y.shape[-1]

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
        tfkl.Dense(DENSITY_PARAMS_SIZE),
        tfpl.MixtureSameFamily(N_C, tfp.layers.MultivariateNormalTriL(O_SIZE))]
        )


# --- TRAIN MODEL ---

LOG_FILE = "log.csv"
CHECKPOINT_FILE = ("checkpoint_epoch_{epoch:02d}/weights")
TRAINED_FILE = "trained/weights"

# Training configuration


def nll(y, Y): return -Y.log_prob(y)


LOSS = nll
BATCH_SIZE = 8192
LEARNING_RATE = 5e-4
EPOCHS = 100
OPTIMISER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
VALIDATION_SPLIT = 0.5

# Callbacks
CSV_LOGGER = cb.CSVLogger(MODEL_DIR + LOG_FILE)
BATCHES_PER_EPOCH = int(np.ceil(X_.shape[0] / BATCH_SIZE
                                * (1 - VALIDATION_SPLIT)))
CHECKPOINTING = cb.ModelCheckpoint(MODEL_DIR + CHECKPOINT_FILE,
                                   save_freq=1 * BATCHES_PER_EPOCH,
                                   verbose=1,
                                   save_weights_only=True)
# EARLY_STOPPING = cb.EarlyStopping(monitor='val_loss', patience=2,
#                                   min_delta=1e-3)
CALLBACKS = [CHECKPOINTING, CSV_LOGGER]  # , EARLY_STOPPING]

# Model compilation and training
model.compile(loss=LOSS, optimizer=OPTIMISER)

History = model.fit(X_, Y_,
                    epochs=EPOCHS,
                    callbacks=CALLBACKS,
                    batch_size=BATCH_SIZE,
                    validation_split=VALIDATION_SPLIT,
                    verbose=2)

model.save_weights(MODEL_DIR + TRAINED_FILE)
