import sys
import os
import time
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cmocean
import regionmask
from shapely.geometry import Point
from shapely.ops import unary_union
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.preprocessing import Scaler
from tools import grids

ccrs = cartopy.crs
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
# kl = tfd.kullback_leibler
tf.keras.backend.set_floatx("float64")
plt.style.use('./misc/poster.mplstyle')
plt.ioff()

# Simulation parameters
N_DRIFTERS = 1000
T = 180  # 65  # Time of simulation in days
REJECT_LAND = True


# =============================================================================
# # model_dir = "models/eddie1904_lr_d10/"
# model_dir = "models/eddie0205_bs8192_lr5em4/"
# figures_dir = model_dir + "figures/"
# # trained_model_file = model_dir + "trained_nn"
# trained_model_file = model_dir + "checkpoint_nn_20_4475.83"
# =============================================================================

# =============================================================================
# if REJECT_LAND:
#     land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
#     land_poly = unary_union(land.polygons)
#     ar6_land = regionmask.defined_regions.ar6.land
#     ar6_land_poly = unary_union(ar6_land.polygons)
#     mask = N_DRIFTERS.fromfile(
#         "data/GDP/masks/EASE2_M36km.LOCImask_land50_coast0km.964x406.bin",
#         dtype=int, sep=""
#     )
# 
# 
# # LOAD MODEL
# 
# DT = 2
# MODEL_DIR = "dx/models/GDP_2day_ml_periodic/"
# CHECKPOINT = "trained"
# DATA_DIR = "data/GDP/2day/"
# 
# X = N_DRIFTERS.load(DATA_DIR + "X0_train.N_DRIFTERSy")
# Y = N_DRIFTERS.load(DATA_DIR + "DX_train.N_DRIFTERSy")
# 
# Xws = X.copy()
# Xws[:, 0] -= 360.
# Xes = X.copy()
# Xes[:, 0] += 360.
# 
# # Periodicising X0.
# X = N_DRIFTERS.concatenate((X, Xes, Xws), axis=0)
# Y = N_DRIFTERS.concatenate((Y, Y, Y), axis=0)
# 
# Xscaler = Scaler(X)
# Yscaler = Scaler(Y)
# X_size = X.shape[0]
# 
# del X, Y, Xws, Xes
# 
# 
# # --- BUILD MODEL ---
# 
# # Data attributes
# O_SIZE = len(Yscaler.mean)
# 
# # Model hyperparameters
# N_C = 32
# 
# DENSITY_PARAMS_SIZE = tfpl.MixtureSameFamily.params_size(
#     N_C, component_params_size=tfpl.MultivariateNormalTriL.params_size(O_SIZE))
# 
# 
# # mirrored_strategy = tf.distribute.MirroredStrategy()
# # with mirrored_strategy.scope():
# model = tf.keras.Sequential([
#     tfkl.Dense(256, activation='relu'),
#     tfkl.Dense(256, activation='relu'),
#     tfkl.Dense(256, activation='relu'),
#     tfkl.Dense(256, activation='relu'),
#     tfkl.Dense(512, activation='relu'),
#     tfkl.Dense(512, activation='relu'),
#     tfkl.Dense(DENSITY_PARAMS_SIZE),
#     tfpl.MixtureSameFamily(N_C, tfpl.MultivariateNormalTriL(O_SIZE))]
# )
# 
# # Load weights
# model.load_weights(MODEL_DIR + CHECKPOINT + "/weights")
# 
# 
# # Configure drifters.
# TN = int(T / DT)
# X = N_DRIFTERS.zeros((N_DRIFTERS, TN + 1, 2))
# X0 = N_DRIFTERS.load("data/GDP/coords/sampled_drifters_inits.npy")[:N_DRIFTERS, :]
# X[:, 0, :] = X0
# 
# # Check X0s don't intersect land poly.
# on_land = N_DRIFTERS.array([Point(X0[i, :]).intersects(land_poly)
#                     for i in range(X0.shape[0])]).sum()
# if on_land > 0:
#     print("X0 on land!")
# 
# 
# # Simulate drifter evolution.
# t0 = time.time()
# if REJECT_LAND:
#     land_proposals = 0
#     maybe_land_proposals = 0
#     for d in range(N_DRIFTERS):
#         print(f"d = {d}")
#         for tn in range(TN):
#             on_land = True
#             while on_land:
#                 proposal = X[d:d + 1, tn, :] + (
#                     Yscaler.invert_standardisation(
#                         model(Xscaler.standardise(
#                             N_DRIFTERS.mod(X[d:d + 1, tn, :] + 180., 360.) - 180.))))
#                 maybe_on_land = Point(
#                     proposal.numpy().flatten()).intersects(ar6_land_poly)
#                 if maybe_on_land:
#                     on_land = Point(
#                         proposal.numpy().flatten()).intersects(land_poly)
#                     maybe_land_proposals += 1
#                 else:
#                     on_land = False
#                 if on_land:
#                     land_proposals += 1
#             X[d:d + 1, tn + 1, :] = proposal
# else:
#     for tn in range(TN):
#         X[:, tn + 1, :] = X[:, tn, :] + (
#             Yscaler.invert_standardisation(
#                 model(Xscaler.standardise(
#                     N_DRIFTERS.mod(X[:, tn, :] + 180., 360.) - 180.))))
# t1 = time.time()
# time_sim = t1 - t0
# print(f"Simulation took {time_sim // 60:.0f} minutes, "
#       + f"{time_sim - 60 * (time_sim // 60):.0f} seconds.")
# 
# 
# np.save("data/GDP/model_simulations/"
#         + "1000simulated_drifters180days_2day_ml_periodic.npy", X)
# 
# =============================================================================
# ---

X = np.load("data/GDP/model_simulations/"
            + "1000simulated_drifters180days_2day_ml_periodic.npy")


plt.figure(figsize=(8.5085, 8.5085 / 2))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
# ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# ax.set_extent([-360., 180., -90., 90.], crs=ccrs.PlateCarree())
ax.set_global()
# plt.title("Drifter trajectories")
for d in range(N_DRIFTERS):
    sca = ax.plot(X[d, :, 0], X[d, :, 1],  # color='k',
                  linewidth=0.5,
                  transform=ccrs.Geodetic())
ax.add_feature(cartopy.feature.LAND, zorder=100,
               facecolor='k')
# ax.coastlines()
plt.tight_layout()
# plt.show()

plt.savefig("figures/1000_simulated_drifters_15_09_22.png",
            format='png')
