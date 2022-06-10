"""
Test regularisation of Gaussian mixture for density estimation.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from dn import Gm
from data_preprocess import Scaler
plt.style.use('./figures/experiments.mplstyle')
tf.keras.backend.set_floatx('float64')
tfd = tfp.distributions


# normal = tfd.Normal(loc=tf.constant(0.5,
#                                     dtype='float64'),
#                     scale=tf.constant())


# Make true Gaussian mixture density.
gm_true = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=tf.constant((0.5, 0.1, 0.05, 0.05, 0.3),
                          dtype='float64')),
    components_distribution=tfd.Normal(
        loc=tf.constant((0., 2., 5., 6., 10.),
                        dtype='float64'),
        scale=tf.constant((.1, .2, .05, .3, .3),
                          dtype='float64')))
gm_true.dtype

x = np.linspace(-5., 15., 1000)
p_true = gm_true.prob(x)

# Sample from true GM.
BATCH_SIZE = 32
N = BATCH_SIZE * 1000
X = gm_true.sample(N).numpy()
Xscaler = Scaler(X)
X_ = Xscaler.normalise(X[:, None])


# Make trivial MDN to fit Gaussian mixture to 1D samples.

# Network hyperparameters
N_C = 24                # Number of mixture components

# Initialise Gaussian mixture "layer"
gm = Gm(o_size=1, n_c=N_C, scale_type='diag', reg_c=(1e-3, None, 1e-1))

# Training parameters
LEARNING_RATE = 1e-2
LOSS = gm.nll_reg
METRICS = [gm.nll, ]
OPTIMISER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)


def initialise_nn():
    """ Initialise neural network. """
    # Create input layer.
    l_i = tf.keras.layers.Input(shape=(1,), name='input_layer',
                                batch_size=BATCH_SIZE)

    # Make output layer.
    l_o = tf.keras.layers.Dense(gm.params_size, kernel_initializer='zeros',
                                use_bias=True,
                                bias_initializer='glorot_uniform',
                                name='params_layer')(l_i)
    # Construct model
    neural_net = tf.keras.models.Model(l_i, l_o)
    neural_net.compile(loss=LOSS, optimizer=OPTIMISER, metrics=METRICS)
    return neural_net


LR_SCHEDULE = tf.keras.callbacks.LearningRateScheduler(
    lambda e: 1e-2 * 10 ** (-float(np.floor(e / 30))))


nn = initialise_nn()
N_EPOCHS = 10  # 0 !!!
History = nn.fit(X_ * 0., X_, epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                 verbose=1, validation_split=0.5, callbacks=[LR_SCHEDULE])
gm.neural_net = nn


# History plots
plt.figure()
plt.plot(range(1, N_EPOCHS + 1), History.history['loss'], 'k',
         label='Training loss')
plt.plot(range(1, N_EPOCHS + 1), History.history['val_loss'], 'r',
         label='Test loss')
plt.plot(range(1, N_EPOCHS + 1), History.history['nll'], 'k--',
         label='Training nll')
plt.plot(range(1, N_EPOCHS + 1), History.history['val_nll'], 'r--',
         label='Test nll')
plt.xlabel('Epoch')
plt.xlabel('Epoch')
# plt.yscale('log')
plt.legend()
plt.tight_layout()


# Reconstruct learned model in original units.
weights, loc, scale = gm.get_params_from_x(np.array((0.,))[None, :])
# loc = Xscaler.mean + Xscaler.std * loc
# scale *= Xscaler.std
loc = Xscaler.invert_normalisation_loc(loc)
scale = Xscaler.invert_normalisation_cov(scale ** 2.) ** 0.5
gm_model = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=weights[0, :]),
    components_distribution=tfd.Normal(
        loc=loc[0, :, 0],
        scale=scale[0, :, 0]))

print('Largest five weights sum to '
      + f'{np.sort(weights.numpy())[0, -5:].sum():.3f}.')
loc5 = np.sort(loc[weights > 0.005].numpy().flatten())
scale5 = np.sort(scale[weights > 0.005].numpy().flatten())

p_model = gm_model.prob(x[:, None])

plt.figure()
plt.plot(x, p_true, 'k')
plt.plot(x, p_model, 'r--')
plt.scatter(X, np.ones_like(X), c='r', marker='.')
plt.grid(True)
plt.yscale('log')
plt.show()


# Validate against true distribution (estimate KL divergence)
XT = gm_true.sample(BATCH_SIZE * 100000)
entropy_trueT = gm_true.log_prob(XT).numpy().mean()
entropy_modelT = gm_model.log_prob(XT).numpy().mean()
KLT = entropy_trueT - entropy_modelT

# Reverse KL to show it isn't symmetric
XM = gm_model.sample(BATCH_SIZE * 100000)
entropy_trueM = gm_true.log_prob(XM).numpy().mean()
entropy_modelM = gm_model.log_prob(XM).numpy().mean()
KLM = entropy_trueM - entropy_modelM
