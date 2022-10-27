import numpy as np
import matplotlib.pyplot as plt
from tools import grids
from dx.mdn_utils import mdn_mean_log_likelihood
import time


# Load data and choose a subset.

DT = 14.
DATA_DIR = f"data/GDP/{DT:.0f}day/"

X0 = np.load(DATA_DIR + "X0_train.npy")
subset_lon = np.logical_and(X0[:, 0] > -52., X0[:, 0] < -10.)
subset_lat = np.logical_and(X0[:, 1] > 31., X0[:, 1] < 60.)
subset = np.logical_and(subset_lon, subset_lat)
X0 = X0[subset]
DX = np.load(DATA_DIR + "DX_train.npy")[subset]

X0val = np.load(DATA_DIR + "X0_test.npy")
subset_lon_val = np.logical_and(X0val[:, 0] > -52., X0val[:, 0] < -10.)
subset_lat_val = np.logical_and(X0val[:, 1] > 31., X0val[:, 1] < 60.)
subset_val = np.logical_and(subset_lon_val, subset_lat_val)
X0val = X0val[subset_val][::100]
DXval = np.load(DATA_DIR + "DX_test.npy")[subset_val][::100]


# Create models.

DTMC = grids.DTMC(n_x=int(360. / 15.), n_y=int(180. / 15.))
GTGP = grids.GTGP(n_x=int(360. / 3.), n_y=int(180. / 3.))


# Fit models.
DTMC.fit(X0, DX)
GTGP.fit(X0, DX)

plt.figure()
plt.pcolormesh(DTMC.vertices[..., 0], DTMC.vertices[..., 1],
               np.log(DTMC.transition_matrix_4d[8, 10]), shading='flat')
# plt.pcolormesh(DTMC.vertices[..., 0], DTMC.vertices[..., 1],
#                DTMC.X0_some_2d, shading='flat')
plt.scatter(DTMC.vertices[8, 10, 0], DTMC.vertices[8, 10, 1],
            marker='x', color='m')
plt.colorbar()
plt.show()

plt.figure()
plt.pcolormesh(GTGP.vertices[..., 0], GTGP.vertices[..., 1],
               GTGP.mean[..., 0].T, shading='flat')
plt.colorbar()
plt.show()


# Score models
DTMC_mll = DTMC.mean_log_likelihood(X0val, DXval)
GTGP_mll = GTGP.mean_log_likelihood(X0val, DXval)


def DTMC_score(res):
    DTMC = grids.DTMC(n_x=int(360. / res), n_y=int(180. / res))
    DTMC.fit(X0, DX)
    return DTMC.mean_log_likelihood(X0val, DXval)


def GTGP_score(res):
    GTGP = grids.GTGP(n_x=int(360. / res), n_y=int(180. / res))
    GTGP.fit(X0, DX)
    return GTGP.mean_log_likelihood(X0val, DXval)


ress = np.logspace(1, 1.4, 10)
DTMC_mll = np.array([DTMC_score(res) for res in ress])

ress = np.logspace(np.log10(2), 1., 20)
# ress = [2., 2.5, 3., 3.5, 4.]
GTGP_mll = np.array([GTGP_score(res) for res in ress])

t0 = time.time()
MDN_mll = mdn_mean_log_likelihood(X0val, DXval,
                                  # "dx/models/GDP_14day_vb_flipout_periodic/",
                                  "dx/models/GDP_14day_NC1_vb/",
                                  DT, 1)
t1 = time.time()
print(f"Took {t1 - t0} seconds")

lnK = (MDN_mll - DTMC_mll) * X0val.shape[0]
lnK = (MDN_mll - GTGP_mll) * X0val.shape[0]
print(f"Bayes factor: {lnK}")


plt.figure()
plt.plot(ress, lnK, 'k*-')
plt.xscale('log')
# plt.yscale('log')
plt.grid(True)
plt.show()
