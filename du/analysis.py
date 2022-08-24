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


MODEL_DIR = "du/models/GLAD_1908/"
# CHECKPOINT = "trained"
CHECKPOINT = "checkpoint_epoch_01"
FIG_DIR = MODEL_DIR + "figures/"
if not Path(FIG_DIR).exists():
    Path(FIG_DIR).mkdir(parents=True)


# --- PREPARE DATA ---

# DATA_DIR = "data/du/"

# X = np.load(DATA_DIR + "r_train.npy")[::10]
# Y = np.load(DATA_DIR + "du_train.npy")[::10]

DATA_DIR = "data/GLAD/"

X = np.load(DATA_DIR + "r.npy")[::1000]
Y = np.load(DATA_DIR + "du.npy")[::1000]

# XVAL = np.load(DATA_DIR + "r_test.npy")
# YVAL = np.load(DATA_DIR + "du_test.npy")

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
    tfkl.Dense(DENSITY_PARAMS_SIZE),
    tfpl.MixtureSameFamily(N_C, tfp.layers.MultivariateNormalTriL(O_SIZE))]
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

# r_range = np.load(DATA_DIR + "du_from_field/r_range.npy")[:, None]
# r = r_range
# r = np.logspace(np.log10(X.min()),
#                 np.log10(X.max()), 100)
base = 1.5
r = np.logspace(np.log(10.) / np.log(base), np.log(1e6) / np.log(base), 100,
                base=base)
gms_ = model(Xscaler.standardise(r[:, None]))
# r = np.logspace(np.log10(r_range[0]),
#                 np.log10(r_range[-1]), 100)
# del X, Y
# gms_ = model(Xscaler.standardise(r))


# --- S2 FIGURE ---

cov = Yscaler.invert_standardisation_cov(gms_.covariance()).numpy()

SF2l = cov[:, 0, 0]
SF2t = cov[:, 1, 1]

plt.figure()
plt.loglog(r, SF2l + SF2t, 'k-', label=r'$S_2(r)$')
plt.plot(r, SF2l, '-', color=cp[5], label=r'$S_2^{(L)}(r)$')
plt.plot(r, SF2t, '-', color=cp[3], label=r'$S_2^{(T)}(r)$')
# plt.plot(np.array([r[0], r[-1]]), 50 * np.array([r[0], r[-1]]) ** 2., '--',
#          color='0.25', label=r'$r^{2}$')
ylims = plt.ylim()
# plt.vlines(2 * np.pi / 64 / 2, ylims[0], ylims[1], 'grey', '-.',
#            label=r'$l_f$')
# plt.vlines(2 * np.pi / 350 / 2, ylims[0], ylims[1], 'grey', ':',
#            label=r'$l_{d}$')
plt.ylim(*ylims)
plt.grid(True)
plt.xlabel(r'$r$')
plt.ylabel(r'Second-order structure functions')
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig(FIG_DIR + "S2.png", dpi=576)


# --- S3 FIGURE ---

def S3(r, sample_size, chunk_size=None):
    if not chunk_size:
        gms = model(r)
        du = gms.sample(sample_shape=(sample_size, ))
        S3l = tf.math.reduce_mean(du[..., 0] ** 3, axis=0)
        S3t = tf.math.reduce_mean(du[..., 0] * du[..., 1] ** 2, axis=0)
    else:
        assert sample_size % chunk_size == 0, "Chunk size must divide sample."
        n_chunks = sample_size // chunk_size
        S3l = np.zeros((n_chunks, r.size))
        S3t = np.zeros((n_chunks, r.size))
        for i in range(n_chunks):
            gms = model(r)
            du = gms.sample((chunk_size, ))
            S3l[i, ...] = (tf.math.reduce_mean(du[..., 0] ** 3, axis=0))
            S3t[i, ...] = tf.math.reduce_mean(
                du[..., 0] * du[..., 1] ** 2, axis=0)
        S3l = tf.math.reduce_mean(S3l, axis=0)
        S3t = tf.math.reduce_mean(S3t, axis=0)
    return S3l, S3t


S3l, S3t = S3(Xscaler.standardise(r), 1000000,  # 0,
              chunk_size=10000)
S3l = S3l * Yscaler.std[0] ** 3
S3t = S3t * Yscaler.std[0] * Yscaler.std[1] ** 2

plt.figure()
plt.loglog(r, S3l + S3t, 'k-', label=r'$V(r)$')
plt.loglog(r, -(S3l + S3t), 'k--', label=r'$-V(r)$')
plt.loglog(r, S3l, 'b-', label=r'$S_{(L)}(r)$')
plt.loglog(r, -S3l, 'b--', label=r'$-S_{(L)}(r)$')
plt.loglog(r, S3t, 'g-', label=r'$S_{(T)}(r)$')
plt.loglog(r, -S3t, 'g--', label=r'$-S_{(T)}(r)$')
plt.plot(np.array([0.01, 0.1]), 1. * np.array([0.01, 0.1]) ** 3, 'g--',
         label=r'$r^{3}$')
plt.vlines(2 * np.pi / 64, 1e-5, 1e-2, 'grey', '-.',
           label=r'$l_f$')
plt.vlines(2 * np.pi / 350, 1e-5, 1e-2, 'grey', ':',
           label=r'$l_{d}$')
plt.ylim(1e-7, None)
plt.xlabel(r'$r$')
plt.legend()
plt.grid()
plt.savefig(FIG_DIR + "S3_loglog.png", dpi=576)
plt.close()


# --- p(du|r) FIGURE ---

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

# rs = np.array([r_range[2], r_range[4], r_range[16]])
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
