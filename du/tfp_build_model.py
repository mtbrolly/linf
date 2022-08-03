"""
Build (and save) Gaussian mixture density network to model spatial velocity
increments.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))


def build_model(model_dir):
    tf.keras.backend.set_floatx("float64")
    mirrored_strategy = tf.distribute.MirroredStrategy()

    # Data parameters
    BATCH_SIZE = 8192
    I_SIZE = 1
    O_SIZE = 2

    # Network hyperparameters
    N_C = 24  # Number of mixture components
    DENSITY_PARAMS_SIZE = tfp.layers.MixtureSameFamily.params_size(
        N_C,
        component_params_size=tfp.layers.MultivariateNormalTriL.params_size(
            O_SIZE,))
    N_HU = [256, 256, 256, 256, 512, 512,
            DENSITY_PARAMS_SIZE]  # Numbers of hidden units
    HL_ACTIVATION = "relu"  # Activation on hidden layer units

    def build_mdn():
        """Construct mixture density network."""
        with mirrored_strategy.scope():
            # Create input layer.
            l_i = tf.keras.layers.Input(
                shape=(I_SIZE), name="input_layer", batch_size=BATCH_SIZE
            )

            # Make hidden layers.
            h_layers = []
            for hl_i in range(len(N_HU)):
                if hl_i == 0:
                    h_layers.append(
                        tf.keras.layers.Dense(
                            N_HU[hl_i],
                            activation=HL_ACTIVATION,
                        )(l_i)
                    )
                elif hl_i < len(N_HU) - 1:
                    h_layers.append(
                        tf.keras.layers.Dense(
                            N_HU[hl_i],
                            activation=HL_ACTIVATION,
                        )(h_layers[-1])
                    )
                else:
                    h_layers.append(
                        tf.keras.layers.Dense(
                            N_HU[hl_i]
                        )(h_layers[-1])
                    )
            # Make Gaussian mixture density layer.
            l_p = tfp.layers.MixtureSameFamily(
                N_C, tfp.layers.MultivariateNormalTriL(O_SIZE),
                name="density_layer")(h_layers[-1])

            # Construct model
            mdn = tf.keras.models.Model(l_i, l_p)
            return mdn

    return build_mdn()
