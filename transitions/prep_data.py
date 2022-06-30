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

    data_dir = "../data/transitions/"

    X = np.load(data_dir + "X0_train.npy")
    XVAL = np.load(data_dir + "X0_test.npy")
    Y = np.load(data_dir + "DX_train.npy")
    YVAL = np.load(data_dir + "DX_test.npy")

    # Reorder data from (lat, lon) to (lon, lat) and deal with displacements
    # which cross the dateline.
    Xsets = [X, XVAL]
    Ysets = [Y, YVAL]

    for i, x in enumerate(Xsets):
        temp = x[:, 1].copy()
        x[:, 1] = x[:, 0]
        x[:, 0] = temp
        del temp, x

    for i, y in enumerate(Ysets):
        temp = y[:, 1].copy()
        y[:, 1] = y[:, 0]
        y[:, 0] = temp
        del temp
        y[:, 0] += (y[:, 0] < -270.) * 360. + (y[:, 0] > 270.) * (-360.)

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
