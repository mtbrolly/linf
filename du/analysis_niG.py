import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from scipy.stats import norm
import seaborn as sns
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


MODEL_DIR = "du/models/niG_1008/"
CHECKPOINT = "trained"
FIG_DIR = MODEL_DIR + "figures/"
if not Path(FIG_DIR).exists():
    Path(FIG_DIR).mkdir(parents=True)


# --- PREPARE DATA ---

DATA_DIR = "data/du/"

X = np.load(DATA_DIR + "r_train.npy")[::10]
Y = np.load(DATA_DIR + "du_train.npy")[::10]

# XVAL = np.load(DATA_DIR + "r_test.npy")
# YVAL = np.load(DATA_DIR + "du_test.npy")

Xscaler = Scaler(X)
Yscaler = Scaler(Y)


# --- BUILD MODEL ---

# Data attributes
O_SIZE = Y.shape[-1]

# Model hyperparameters

DENSITY_PARAMS_SIZE = 8

model = tf.keras.Sequential([
    tfkl.Dense(256, activation='relu'),
    tfkl.Dense(256, activation='relu'),
    tfkl.Dense(256, activation='relu'),
    tfkl.Dense(256, activation='relu'),
    tfkl.Dense(DENSITY_PARAMS_SIZE),
    tfpl.DistributionLambda(
        lambda t: tfd.Independent(
            tfd.NormalInverseGaussian(
                loc=t[..., :O_SIZE],
                scale=tfa.softplus(t[..., O_SIZE:2 * O_SIZE]),
                tailweight=tfa.softplus(t[..., 2 * O_SIZE:3 * O_SIZE]),
                skewness=t[..., 3 * O_SIZE:]),
            reinterpreted_batch_ndims=1))])

# DENSITY_PARAMS_SIZE = 6

# # mirrored_strategy = tf.distribute.MirroredStrategy()
# # with mirrored_strategy.scope():
# model = tf.keras.Sequential([
#     tfkl.Dense(256, activation='relu'),
#     tfkl.Dense(256, activation='relu'),
#     tfkl.Dense(256, activation='relu'),
#     tfkl.Dense(256, activation='relu'),
#     tfkl.Dense(DENSITY_PARAMS_SIZE),
#     tfpl.DistributionLambda(
#         lambda t: tfd.Independent(
#             tfd.StudentT(
#                 loc=t[..., :O_SIZE],
#                 scale=tfa.softplus(t[..., O_SIZE:2 * O_SIZE]),
#                 df=tfa.softplus(t[..., 2 * O_SIZE:3 * O_SIZE])),
#             reinterpreted_batch_ndims=1))])


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

r_range = np.load(DATA_DIR + "du_from_field/r_range.npy")[:, None]
r = r_range
# r = np.logspace(np.log10(r_range[0]),
#                 np.log10(r_range[-1]), 100)
# del X, Y
gms_ = model(Xscaler.standardise(r))


# --- S2 FIGURE ---

var = Yscaler.std ** 2 * gms_.distribution.variance().numpy()
SF2l = var[:, 0]
SF2t = var[:, 1]

plt.figure()
plt.loglog(r, SF2l + SF2t, 'k-*', label=r'$S_2(r)$')
plt.plot(r, SF2l, '-*', color=cp[5], label=r'$S_2^{(L)}(r)$')
plt.plot(r, SF2t, '-*', color=cp[3], label=r'$S_2^{(T)}(r)$')
plt.plot(np.array([r[0], r[-1]]), 50 * np.array([r[0], r[-1]]) ** 2., '--',
         color='0.25', label=r'$r^{2}$')
ylims = plt.ylim()
plt.vlines(2 * np.pi / 64 / 2, ylims[0], ylims[1], 'grey', '-.',
           label=r'$l_f$')
plt.vlines(2 * np.pi / 350 / 2, ylims[0], ylims[1], 'grey', ':',
           label=r'$l_{d}$')
plt.ylim(*ylims)
plt.grid(True)
plt.xlabel(r'$r$')
plt.ylabel(r'Second-order structure functions')
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig(FIG_DIR + "S2.png", dpi=576)

# --- S3 FIGURE ---

# To do!

# --- p(du|r) FIGURE ---

fig, ax = plt.subplots(3, 2, figsize=(10, 10))

rs = np.array([r_range[2], r_range[4], r_range[16]])
for i in range(len(rs)):
    mr_ = model(Xscaler.standardise(rs[i:i + 1]))
    dul_std, dut_std = (
        Yscaler.std * mr_.distribution.stddev().numpy().flatten())
    dul = np.linspace(-5. * dul_std, 5. * dul_std, 200)
    dut = np.linspace(-5. * dut_std, 5. * dut_std, 200)
    du_ = Yscaler.standardise(
        np.concatenate([dul[:, None], dut[:, None]], axis=1))
    pdul = np.exp(
        mr_.distribution[0, 0].log_prob(du_[:, 0]).numpy()) / Yscaler.std[0]
    pdut = np.exp(
        mr_.distribution[0, 1].log_prob(du_[:, 1]).numpy()) / Yscaler.std[1]
    ax[i, 0].plot(dul, np.log(pdul), 'g')
    ax[i, 1].plot(dut, np.log(pdut), 'g')

    # Gaussian densities
    lp_nl = norm.logpdf(dul, loc=0., scale=dul_std)
    lp_nt = norm.logpdf(dut, loc=0., scale=dut_std)
    ax[i, 0].plot(dul, lp_nl, '--', color='0.5')
    ax[i, 1].plot(dut, lp_nt, '--', color='0.5')

    # Add histogram from training data.
    r_ind = (X == rs[i:i + 1])[:, 0]
    du_r = Y[r_ind, :]
    hist, bin_e = np.histogram(du_r[:, 0], bins=dul, density=True)
    ax[i, 0].plot((bin_e[1:] + bin_e[:-1]) / 2, np.log(hist), 'k')
    hist, bin_e = np.histogram(du_r[:, 1], bins=dut, density=True)
    ax[i, 1].plot((bin_e[1:] + bin_e[:-1]) / 2, np.log(hist), 'k')

    ax[i, 0].set_title(rf"$r = {rs[i][0]:.3f}$")
    ax[i, 1].set_title(rf"$r = {rs[i][0]:.3f}$")
    ax[i, 0].set_xlabel(r"$\delta u_{L}$")
    ax[i, 1].set_xlabel(r"$\delta u_{T}$")
    ax[i, 0].set_ylabel(r"$p(\delta u_{L}|r)$")
    ax[i, 1].set_ylabel(r"$p(\delta u_{T}|r)$")
    ax[i, 0].grid(True)
    ax[i, 1].grid(True)
    ax[i, 0].set_ylim(np.log(pdul).min() - 1., None)
    ax[i, 1].set_ylim(np.log(pdut).min() - 1., None)

fig.tight_layout()
fig.show()

plt.savefig(FIG_DIR + "densities.png", dpi=576)
