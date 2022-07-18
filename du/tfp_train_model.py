import pickle
import tensorflow as tf
from tensorflow.keras import callbacks as cb
import numpy as np
import pandas as pd
from tfp_build_model import build_model


def train_model(model_dir):

    tf.keras.backend.set_floatx('float64')
    checkpoint_model_file = (
        model_dir + "checkpoint_epoch_{epoch:02d}/weights")
    trained_model_dir = model_dir + "trained/"
    log_file = "log.csv"
    history_file = "history.csv"

    # Generate mixture density network.
    MDN = build_model(model_dir)

    # Load data.
    datas = []
    datas_str = ["X", "XVAL", "Y", "YVAL"]
    for i in range(len(datas_str)):
        datas.append(np.load(model_dir + datas_str[i] + '.npy'))

    scalers = []
    scalers_str = ["Xscaler", "Yscaler"]
    for i in range(len(scalers_str)):
        with open(model_dir + scalers_str[i] + '.pickle', 'rb') as f:
            scalers.append(pickle.load(f))

    [Xscaler, Yscaler] = scalers

    [X, XVAL, Y, YVAL] = datas
    X_ = Xscaler.standardise(X)
    del X
    XVAL_ = Xscaler.standardise(XVAL)
    del XVAL
    Y_ = Yscaler.standardise(Y)
    del Y
    YVAL_ = Yscaler.standardise(YVAL)

    # Data attributes
    BATCH_SIZE = MDN.layers[0].input_shape[0][0]

    # Training parameters
    def negloglik(y, rv_y): return -rv_y.log_prob(y)
    LOSS = negloglik
    METRICS = None
    LEARNING_RATE = 5e-4  # !!!
    OPTIMISER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    EPOCHS = 100  # !!!

    BATCHES_PER_EPOCH = int(X_.shape[0] / BATCH_SIZE)
    CHECKPOINTING = cb.ModelCheckpoint(checkpoint_model_file, monitor='loss',
                                       save_freq=1 * BATCHES_PER_EPOCH,
                                       verbose=1,
                                       save_weights_only=True)
    CSV_LOGGER = cb.CSVLogger(model_dir + log_file)
    CALLBACKS = [CHECKPOINTING,
                 CSV_LOGGER]

    # Compile and train model
    MDN.compile(loss=LOSS, optimizer=OPTIMISER, metrics=METRICS)
    History = MDN.fit(X_, Y_, initial_epoch=0, epochs=EPOCHS,
                      callbacks=CALLBACKS, batch_size=BATCH_SIZE,
                      validation_data=[XVAL_, YVAL_],
                      verbose=1)

    # Save training history and trained model.
    hist_df = pd.DataFrame(History.history)
    with open(model_dir + history_file, mode='w') as f:
        hist_df.to_csv(f)

    MDN.save_weights(trained_model_dir + '/weights')
