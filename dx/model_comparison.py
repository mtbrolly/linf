"""
Script for computing scores of MDN, DTMC and GTGP models of transition density.
"""

import numpy as np
import matplotlib.pyplot as plt
from tools import grids
from dx.mdn_utils2 import mdn_mean_log_likelihood
plt.style.use('./misc/paper.mplstyle')
plt.ioff()


# Load data and choose a subset.

DT = 4.
DATA_DIR = f"data/GDP/{DT:.0f}day/"

# Set region limits.

global_ = np.array([[-180., 180.], [-90., 90.]])
A = np.array([[-50., -20.], [30., 50.]])
B = np.array([[145., 175.], [20., 40.]])
C = np.array([[-130., -100.], [-10., 10.]])

region = B

X0 = np.load(DATA_DIR + "X0_train.npy")
DX = np.load(DATA_DIR + "DX_train.npy")

# Get X1 from DX.
X1 = X0.copy() + DX.copy()
X1[(X1 > 180)[:, 0], 0] -= 360.
X1[(X1 < -180)[:, 0], 0] += 360.

# Extract subset.

subset_lon_0 = np.logical_and(X0[:, 0] > region[0, 0], X0[:, 0] < region[0, 1])
subset_lat_0 = np.logical_and(X0[:, 1] > region[1, 0], X0[:, 1] < region[1, 1])

subset_lon_1 = np.logical_and(X1[:, 0] > region[0, 0], X1[:, 0] < region[0, 1])
subset_lat_1 = np.logical_and(X1[:, 1] > region[1, 0], X1[:, 1] < region[1, 1])

subset = np.logical_and(
    np.logical_and(
        np.logical_and(
            subset_lon_0, subset_lat_0), subset_lon_1), subset_lat_1)

X0 = X0[subset]
DX = DX[subset]


X0val = np.load(DATA_DIR + "X0_test.npy")
DXval = np.load(DATA_DIR + "DX_test.npy")

# Get X1val from DXval.
X1val = X0val.copy() + DXval.copy()
X1val[(X1val > 180)[:, 0], 0] -= 360.
X1val[(X1val < -180)[:, 0], 0] += 360.

subset_lon_0_val = np.logical_and(X0val[:, 0] > region[0, 0],
                                  X0val[:, 0] < region[0, 1])
subset_lat_0_val = np.logical_and(X0val[:, 1] > region[1, 0],
                                  X0val[:, 1] < region[1, 1])

subset_lon_1_val = np.logical_and(X1val[:, 0] > region[0, 0],
                                  X1val[:, 0] < region[0, 1])
subset_lat_1_val = np.logical_and(X1val[:, 1] > region[1, 0],
                                  X1val[:, 1] < region[1, 1])

subset_val = np.logical_and(
    np.logical_and(
        np.logical_and(
            subset_lon_0_val,
            subset_lat_0_val),
        subset_lon_1_val),
    subset_lat_1_val)

X0val = X0val[subset_val]
DXval = np.load(DATA_DIR + "DX_test.npy")[subset_val]

del X1, X1val


# Compute model scores.


def DTMC_score(res):
    DTMC = grids.DTMC(n_x=int(np.ceil((region[0, 1] - region[0, 0]) / res)),
                      n_y=int(np.ceil((region[1, 1] - region[1, 0]) / res)),
                      xlims=region[0], ylims=region[1])
    DTMC.fit(X0, DX)
    return (DTMC.mean_log_likelihood(X0, DX),
            DTMC.mean_log_likelihood(X0val, DXval))


ress = np.arange(5., 15.)  # Global GTGP (7 gives best val. sco.)
# ress = [45., 60., 90., 180.]  # Global DTMC (only 180 gives finite val. sco.)

# ress = np.logspace(np.log10(.25), np.log10(20), 15)  # A

# ress = np.logspace(np.log10(.25), np.log10(20), 15)  # B (DTMC:10, GTGP:2.5)

# ress = np.logspace(np.log10(1.5), np.log10(20), 12)  # C (DTMC:10)
# ress = np.logspace(np.log10(1.5), np.log10(6), 12)  # C (GTGP:4.11)


DTMC_mll = []
for i in range(len(ress)):
    DTMC_mll.append(DTMC_score(ress[i]))
    print(i + 1)

DTMC_mll = np.array(DTMC_mll)


def GTGP_score(res):
    GTGP = grids.GTGP(n_x=int(np.ceil((region[0, 1] - region[0, 0]) / res)),
                      n_y=int(np.ceil((region[1, 1] - region[1, 0]) / res)),
                      xlims=region[0], ylims=region[1])
    GTGP.fit(X0, DX)
    return (GTGP.mean_log_likelihood(X0, DX),
            GTGP.mean_log_likelihood(X0val, DXval))


GTGP_mll = []
for i in range(len(ress)):
    GTGP_mll.append(GTGP_score(ress[i]))
    print(i + 1)
    print(GTGP_mll[-1])

GTGP_mll = np.array(GTGP_mll)


MDN_mll = mdn_mean_log_likelihood(X0val, DXval,
                                  DT, 32,
                                  block_size=40000)
