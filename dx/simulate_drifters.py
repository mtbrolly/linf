"""
Script for simulating drifters from uniform grid of initial positions using the
MDN model.
"""

import sys
import os
import time
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pickle
from shapely.geometry import Point
from dx.utils import load_mdn, load_scalers
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.preprocessing import Scaler  # noqa: E402
from tools import grids  # noqa: E402

tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
kl = tfd.kullback_leibler
tf.keras.backend.set_floatx("float64")

# Simulation parameters
T = 10 * 365  # Time of simulation in days
REJECT_LAND = True

with open('./data/GDP/masks/land_poly.pkl', "rb") as poly_file:
    land_poly = pickle.load(poly_file)
with open('./data/GDP/masks/ar6_land_poly.pkl', "rb") as poly_file:
    ar6_land_poly = pickle.load(poly_file)


# Model hyperparameters
N_C = 32
DT = 4

MODEL_DIR = (f"dx/models/GDP_{DT:.0f}day_NC{N_C}/")

model = load_mdn(DT=DT, N_C=N_C)
Xscaler, Yscaler = load_scalers(DT=DT, N_C=N_C)

print("Model loaded.")

# Configure drifters -- delete those on land.
TN = int(T / DT)
X0_grid = grids.LonlatGrid(n_x=180, n_y=90)
X0 = X0_grid.centres.reshape((-1, 2))
X0_on_land = np.array([False, ] * len(X0))
for d in range(len(X0)):
    print(d)
    if Point(X0[d]).intersects(land_poly):
        X0_on_land[d] = True
X0 = X0[~X0_on_land]
del X0_on_land
X = np.zeros((len(X0), TN + 1, 2))
X[:, 0, :] = X0


# Simulate drifter evolution.
t0 = time.time()

for tn in range(TN):
    print(f"t = {tn * DT:.0f} days")
    t00 = time.time()

    proposals_on_land = np.array([True, ] * len(X))
    proposals = np.zeros((len(X), 2))

    while proposals_on_land.sum() > 0:
        proposals[proposals_on_land, :] = np.mod(
            X[proposals_on_land, tn, :] + (Yscaler.invert_standardisation(
                model(
                    Xscaler.standardise(np.mod(
                        X[proposals_on_land, tn, :] + 180., 360.) - 180.))))
            + 180., 360.) - 180.

        for d in range(len(X)):
            if proposals_on_land[d]:
                if not Point(proposals[d]).intersects(land_poly):
                    proposals_on_land[d] = False
        # Also check latitude not beyond min/max.
        proposals_on_land[
           ~((proposals[:, 1] > -90.) * (proposals[:, 1] < 90.))] = True
        print(proposals_on_land.sum())

    X[:, tn + 1, :] = proposals

    t01 = time.time()
    print(f"Timestep took {t01 - t00:.0f} seconds.")
t1 = time.time()
time_sim = t1 - t0
print(f"Simulation took {time_sim // 60:.0f} minutes, "
      + f"{time_sim - 60 * (time_sim // 60):.0f} seconds.")


np.save(MODEL_DIR + "homogeneous_release_10year_rejection.npy", X)
print("success")
