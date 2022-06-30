"""
Prepare Global Drifter Program data for training.
"""

# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.preprocessing import Scaler  # noqa: E402


def prep_data(model_dir):
    tf.keras.backend.set_floatx('float64')

    model_file = model_dir + "untrained_nn"
    NN = tf.keras.models.load_model(model_file)
    BATCH_SIZE = NN.layers[0].input_shape[0][0]

    data_dir = "data/velocity_differences/"

    n_skip = 1
    X = np.load(data_dir + "r_train.npy")[::n_skip, :]
    XVAL = np.load(data_dir + "r_test.npy")[::n_skip, :]
    Y = np.load(data_dir + "du_train.npy")[::n_skip, :]
    YVAL = np.load(data_dir + "du_test.npy")[::n_skip, :]

    # Shuffle all data.
    rng = np.random.default_rng(seed=1)
    rng.shuffle(X)
    rng = np.random.default_rng(seed=1)
    rng.shuffle(XVAL)
    rng = np.random.default_rng(seed=1)
    rng.shuffle(Y)
    rng = np.random.default_rng(seed=1)
    rng.shuffle(YVAL)

    # Truncate data to be divisible by batch_size / train_size.
    N = int(min(X.shape[0], XVAL.shape[0]) // BATCH_SIZE) * BATCH_SIZE
    X = X[:N, :]
    XVAL = XVAL[:N, :]
    Y = Y[:N, :]
    YVAL = YVAL[:N, :]

    # Standardise data.
    Xscaler = Scaler(X)
    X_ = Xscaler.standardise(X)
    X_VAL = Xscaler.standardise(XVAL)
    Yscaler = Scaler(Y)
    Y_ = Yscaler.standardise(Y)
    Y_VAL = Yscaler.standardise(YVAL)

    datas = [X, X_, XVAL, X_VAL, Y, Y_, YVAL, Y_VAL]
    datas_str = ["X", "X_", "XVAL", "X_VAL", "Y", "Y_", "YVAL", "Y_VAL"]
    for i, data in enumerate(datas):
        np.save(model_dir + datas_str[i] + '.npy', data)

    scalers = [Xscaler, Yscaler]
    scalers_str = ["Xscaler", "Yscaler"]

    for i, s in enumerate(scalers):
        with open(model_dir + scalers_str[i] + '.pickle', 'wb') as f:
            pickle.dump(s, f, pickle.HIGHEST_PROTOCOL)
