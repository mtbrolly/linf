import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path
from scipy.stats import norm
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.preprocessing import Scaler  # noqa: E402
tfd = tfp.distributions
tfa = tf.keras.activations
tf.keras.backend.set_floatx("float64")
plt.style.use('./misc/experiments.mplstyle')
cp = sns.color_palette("husl", 8)
plt.ioff()

tfkl = tf.keras.layers
tfpl = tfp.layers


MODEL_DIR = "du/models/GDP/"
CHECKPOINT = "trained"
# CHECKPOINT = "checkpoint_epoch_01"
FIG_DIR = MODEL_DIR + "figures/"
if not Path(FIG_DIR).exists():
    Path(FIG_DIR).mkdir(parents=True)


# --- PREPARE DATA ---

DATA_DIR = "data/dx/"

X = np.load(DATA_DIR + "X0_train.npy")
Y = np.load(DATA_DIR + "DX_train.npy")

Xscaler = Scaler(X)
Yscaler = Scaler(Y)


# --- BUILD MODEL ---

# Data attributes
O_SIZE = Y.shape[-1]

# Model hyperparameters
N_C = 32

DENSITY_PARAMS_SIZE = tfpl.MixtureSameFamily.params_size(
    N_C, component_params_size=tfpl.MultivariateNormalTriL.params_size(O_SIZE))

# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():
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

# Load weights
model.load_weights(MODEL_DIR + CHECKPOINT + "/weights")

# Load log
history = pd.read_csv(MODEL_DIR + "log.csv")


# --- LOSS PLOT ---

plt.figure()
plt.plot(range(1, len(history) + 1), history['loss'], 'k',
         label='Training loss')
plt.plot(range(1, len(history) + 1), history['val_loss'], 'grey',
         label='Test loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(FIG_DIR + "loss.png", dpi=576)
plt.close()


# --- CALCULATIONS ---

r = np.logspace(np.log10(X.min()),
                np.log10(X.max()), 100)
gms_ = model(Xscaler.standardise(r[:, None]))


# --- p(DX|X0) FIGURE ---  # !!! Needs edited.

def get_marg(gms, y_ind):
    """
    Takes a multivariate Gaussian mixture which models y|x and y_ind (index
    of a component of y) and returns the marginal (scalar) Gaussian mixture
    which models y[y_ind]|x.
    """
    marg_loc = gms.components_distribution.loc[..., y_ind]
    marg_scale = gms.components_distribution.covariance()[...,
                                                          y_ind, y_ind]
    marginal_gms = tfd.MixtureSameFamily(
        mixture_distribution=gms.mixture_distribution,
        components_distribution=tfd.Normal(
            loc=marg_loc,
            scale=marg_scale))
    return marginal_gms


fig, ax = plt.subplots(3, 2, figsize=(10, 10))

rs = np.array([10 ** 3, 10 ** 4, 10 ** 5])[:, None]
for i in range(len(rs)):
    # Model pdf
    mr = model(Xscaler.standardise(rs[i:i + 1]))
    dul_std, dut_std = Yscaler.invert_standardisation_cov(
        mr.covariance().numpy().squeeze()).diagonal() ** 0.5
    dul = np.linspace(-5. * dul_std, 5. * dul_std, 200)
    dut = np.linspace(-5. * dut_std, 5. * dut_std, 200)
    du_ = Yscaler.standardise(
        np.concatenate([dul[:, None], dut[:, None]], axis=1))
    pdul = np.exp(get_marg(mr, 0).log_prob(du_[:, 0]).numpy()) / Yscaler.std[0]
    pdut = np.exp(get_marg(mr, 1).log_prob(du_[:, 1]).numpy()) / Yscaler.std[1]
    ax[i, 0].plot(dul, np.log(pdul), 'g')
    ax[i, 1].plot(dut, np.log(pdut), 'g')

    # Gaussian pdf with same variance
    lp_nl = norm.logpdf(dul, loc=0., scale=dul_std)
    lp_nt = norm.logpdf(dut, loc=0., scale=dut_std)
    ax[i, 0].plot(dul, lp_nl, '--', color='0.5')
    ax[i, 1].plot(dut, lp_nt, '--', color='0.5')

    # Data histogram
    r_ind = (X == rs[i: i + 1])[:, 0]
    du_r = Y[r_ind, :]
    hist, bin_e = np.histogram(du_r[:, 0], bins=dul, density=True)
    ax[i, 0].plot((bin_e[1:] + bin_e[:-1]) / 2, np.log(hist), 'k')
    hist, bin_e = np.histogram(du_r[:, 1], bins=dut, density=True)
    ax[i, 1].plot((bin_e[1:] + bin_e[:-1]) / 2, np.log(hist), 'k')

    ax[i, 0].set_title(rf"$r = {rs[i, 0]:.3f}$")
    ax[i, 1].set_title(rf"$r = {rs[i, 0]:.3f}$")
    ax[i, 0].set_xlabel(r"$\delta u_{L}$")
    ax[i, 1].set_xlabel(r"$\delta u_{T}$")
    ax[i, 0].set_ylabel(r"$\log p(\delta u_{L}|r)$")
    ax[i, 1].set_ylabel(r"$\log p(\delta u_{T}|r)$")
    ax[i, 0].grid(True)
    ax[i, 1].grid(True)
    ax[i, 0].set_ylim(np.log(pdul).min() - 1., None)
    ax[i, 1].set_ylim(np.log(pdut).min() - 1., None)

fig.tight_layout()
fig.show()

plt.savefig(FIG_DIR + "densities.png", dpi=576)
