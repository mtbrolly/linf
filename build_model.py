"""
Build (and save) Gaussian mixture density network to model transition density
of drifters.
"""

import tensorflow as tf
import pickle
from dn import Gm


def build_model(model_dir):
    tf.keras.backend.set_floatx('float64')
    mirrored_strategy = tf.distribute.MirroredStrategy()

    model_file = model_dir + "untrained_nn"

    # Data parameters
    BATCH_SIZE = 8192
    I_SIZE = 2
    O_SIZE = 2

    # Network hyperparameters
    N_C = 24                                # Number of mixture components
    N_HU = [256, 256, 256, 256, 512, 512]   # Numbers of hidden units
    HL_ACTIVATION = 'relu'                  # Activation on hidden layer units
    HL_REGULARISER = tf.keras.regularizers.L2(l2=0.1)  # H. layer weight reg.

    # Initialise Gaussian mixture "layer"
    gm = Gm(o_size=O_SIZE, n_c=N_C, scale_type='tril')

    def build_nn():
        """ Initialise neural network. """
        with mirrored_strategy.scope():
            # Create input layer.
            l_i = tf.keras.layers.Input(shape=(I_SIZE), name='input_layer',
                                        batch_size=BATCH_SIZE)

            # Make hidden layers.
            h_layers = []
            for hl_i in range(len(N_HU)):
                if hl_i == 0:
                    h_layers.append(tf.keras.layers.Dense(
                        N_HU[hl_i], activation=HL_ACTIVATION,
                        kernel_regularizer=HL_REGULARISER,
                        name='hidden_layer_{}'.format(hl_i))(l_i))
                else:
                    h_layers.append(tf.keras.layers.Dense(
                        N_HU[hl_i], activation=HL_ACTIVATION,
                        name='hidden_layer_{}'.format(hl_i))(h_layers[-1]))
            # Make output layer.
            l_o = tf.keras.layers.Dense(gm.params_size, use_bias=False,
                                        name='params_layer')(h_layers[-1])

            # Construct model
            neural_net = tf.keras.models.Model(l_i, l_o)
            return neural_net

    nn = build_nn()
    nn.save(model_file)

    with open(model_dir + 'gm.pickle', 'wb') as f:
        pickle.dump(gm, f, pickle.HIGHEST_PROTOCOL)
