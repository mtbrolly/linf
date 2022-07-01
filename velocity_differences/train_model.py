import pickle
import tensorflow as tf
from tensorflow.keras import callbacks as cb
import numpy as np
import pandas as pd


def train_model(model_dir):
    tf.keras.backend.set_floatx('float64')

    untrained_model_file = model_dir + "untrained_nn"
    checkpoint_model_file = (model_dir + "checkpoint_epoch_{epoch:02d}")
    trained_model_file = model_dir + "trained_nn"

    # Load neural network and Gaussian mixture layer.
    NN = tf.keras.models.load_model(untrained_model_file)
    with open(model_dir + 'gm.pickle', 'rb') as f:
        gm = pickle.load(f)
    gm.neural_net = NN

    # Load data.
    datas = []
    datas_str = ["X_", "X_VAL", "Y_", "Y_VAL"]
    for i in range(len(datas_str)):
        datas.append(np.load(model_dir + datas_str[i] + '.npy'))

    [X_, X_VAL, Y_, Y_VAL] = datas

    # Data attributes
    BATCH_SIZE = NN.layers[0].input_shape[0][0]

    # Training parameters
    LOSS = gm.nll_reg
    METRICS = None
    LEARNING_RATE = 5e-6
    OPTIMISER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    EPOCHS = 40  # !!!

    BATCHES_PER_EPOCH = int(X_.shape[0] / BATCH_SIZE)
    CHECKPOINTING = cb.ModelCheckpoint(checkpoint_model_file, monitor='loss',
                                       save_freq=10 * BATCHES_PER_EPOCH,
                                       verbose=1)
    CSV_LOGGER = cb.CSVLogger(model_dir + "log.csv")
    CALLBACKS = [CHECKPOINTING, CSV_LOGGER]

    # Compile and train model
    NN.compile(loss=LOSS, optimizer=OPTIMISER, metrics=METRICS)
    History = NN.fit(X_, Y_, initial_epoch=0, epochs=EPOCHS,
                     callbacks=CALLBACKS, batch_size=BATCH_SIZE,
                     validation_data=[X_VAL, Y_VAL],
                     verbose=1)

    # Save training history and trained model.
    hist_df = pd.DataFrame(History.history)
    with open(model_dir + "history.csv", mode='w') as f:
        hist_df.to_csv(f)

    NN.save(trained_model_file)
