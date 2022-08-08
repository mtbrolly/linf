"""
Prepare Global Drifter Program data for training.
"""

# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import sys
import os
from tfp_build_model import build_model
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.preprocessing import Scaler  # noqa: E402

model_name = 'test_tfp'
model_dir = "du/models/" + model_name + "/"


def prep_data(model_dir):
    """
    Prepares data for training:
        - Truncates data to be multiple of batch-size after
        train--test split;
        - Makes and pickles Scaler objects for standardising data.
    """
    tf.keras.backend.set_floatx('float64')

    MDN = build_model(model_dir)
    BATCH_SIZE = MDN.layers[0].input_shape[0][0]

    data_dir = "data/du/"

    X = np.load(data_dir + "r_train.npy")
    XVAL = np.load(data_dir + "r_test.npy")
    Y = np.load(data_dir + "du_train.npy")
    YVAL = np.load(data_dir + "du_test.npy")

    # Shuffle all data.
    # Should be already shuffled...
    # rng = np.random.default_rng(seed=1)
    # rng.shuffle(X)
    # rng = np.random.default_rng(seed=1)
    # rng.shuffle(XVAL)
    # rng = np.random.default_rng(seed=1)
    # rng.shuffle(Y)
    # rng = np.random.default_rng(seed=1)
    # rng.shuffle(YVAL)

    # Truncate data to be divisible by batch_size / train_size.
    N = int(min(X.shape[0], XVAL.shape[0]) // BATCH_SIZE) * BATCH_SIZE
    X = X[:N, :]
    XVAL = XVAL[:N, :]
    Y = Y[:N, :]
    YVAL = YVAL[:N, :]

    # Standardise data.
    Xscaler = Scaler(X)
    Yscaler = Scaler(Y)
    datas = [X, XVAL, Y, YVAL]
    datas_str = ["X", "XVAL", "Y", "YVAL"]
    for i, data in enumerate(datas):
        np.save(model_dir + datas_str[i] + '.npy', data)

    scalers = [Xscaler, Yscaler]
    scalers_str = ["Xscaler", "Yscaler"]

    for i, s in enumerate(scalers):
        with open(model_dir + scalers_str[i] + '.pickle', 'wb') as f:
            pickle.dump(s, f, pickle.HIGHEST_PROTOCOL)
