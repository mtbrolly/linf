import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cmocean
import pickle
import regionmask
from shapely.geometry import Point, LineString
from shapely.ops import unary_union

# Simulation parameters
N_DRIFTERS = 1000
T = 180  # 65  # Time of simulation in days
REJECT_LAND = True

tf.keras.backend.set_floatx('float64')
plt.style.use('./figures/posters.mplstyle')

# model_dir = "models/eddie1904_lr_d10/"
model_dir = "models/eddie0205_bs8192_lr5em4/"
figures_dir = model_dir + "figures/"
# trained_model_file = model_dir + "trained_nn"
trained_model_file = model_dir + "checkpoint_nn_20_4475.83"

if REJECT_LAND:
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_poly = unary_union(land.polygons)
    ar6_land = regionmask.defined_regions.ar6.land
    ar6_land_poly = unary_union(ar6_land.polygons)
    mask = np.fromfile(
        "data/EASE2_M36km.LOCImask_land50_coast0km.964x406.bin",
        dtype=int, sep=""
    )

# Load neural network and Gaussian mixture layer.
with open(model_dir + 'gm.pickle', 'rb') as f:
    gm = pickle.load(f)
NN = tf.keras.models.load_model(trained_model_file,
                                custom_objects={'nll_reg': gm.nll_reg})
gm.neural_net = NN

# Load scalers.
scalers = []
scalers_str = ["Xscaler", "Yscaler"]
for i in range(len(scalers_str)):
    with open(model_dir + scalers_str[i] + '.pickle', 'rb') as f:
        scalers.append(pickle.load(f))
Xscaler, Yscaler = scalers

# Configure drifters.
DT = 2.
TN = int(T / DT)
X = np.zeros((N_DRIFTERS, TN + 1, 2))
locations = {'gulf_stream': [-77.08, 32.71], 'irish_sea': [-5.2, 53.0],
             'gulf_of_mexico': [-90., 25.], 'gulf_of_guinea': [0., 0.]}
# Point of release of drifters in (lon, lat).
# X[:, 0, :] = locations['gulf_of_guinea']

# X0 = np.load("data/drogued_drifters_data/X0_train.npy"
#              )[:, [1, 0]][::1000, ...][:X.shape[0], :]
X0 = np.load("data/sampled_drifters_inits.npy")
X[:, 0, :] = X0

# Check X0s don't intersect land poly.
on_land = np.array([Point(X0[i, :]).intersects(land_poly)
                    for i in range(X0.shape[0])]).sum()
if on_land > 0:
    print("X0 on land!")

# Simulate drifter evolution.
t0 = time.time()
if REJECT_LAND:
    land_proposals = 0
    maybe_land_proposals = 0
    for d in range(N_DRIFTERS):
        print(f"d = {d}")
        for tn in range(TN):
            on_land = True
            while on_land:
                proposal = X[d:d + 1, tn, :] + (
                    Yscaler.invert_standardisation(
                        gm.sample(Xscaler.standardise(
                            np.mod(X[d:d + 1, tn, :] + 180., 360.) - 180.))))
                maybe_on_land = Point(
                    proposal.numpy().flatten()).intersects(ar6_land_poly)
                if maybe_on_land:
                    on_land = Point(
                        proposal.numpy().flatten()).intersects(land_poly)
                    maybe_land_proposals += 1
                else:
                    on_land = False
                # on_land = LineString([X[d:d + 1, tn, :].flatten(),
                #                       proposal.numpy().flatten()]).intersects(
                #                           land_poly.boundary)
                if on_land:
                    land_proposals += 1
            X[d:d + 1, tn + 1, :] = proposal
else:
    for tn in range(TN):
        X[:, tn + 1, :] = X[:, tn, :] + (
            Yscaler.invert_standardisation(
                gm.sample(Xscaler.standardise(
                    np.mod(X[:, tn, :] + 180., 360.) - 180.))))
t1 = time.time()
time_sim = t1 - t0
print(f"Simulation took {time_sim // 60:.0f} minutes, "
      + f"{time_sim - 60 * (time_sim // 60):.0f} seconds.")

# np.save("data/1000simulated_drifters180.npy", X)
# X = np.load("data/1000simulated_drifters.npy")

# Plot drifter trajectories.
plt.ioff()
plt.figure(figsize=(15, 7.5))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
ax.set_extent([-180., 180., -90., 90.], crs=ccrs.PlateCarree())
# ax = plt.axes(projection=ccrs.Orthographic(central_latitude=-90.))
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
plt.title("Simulated drifter trajectories")
for d in range(N_DRIFTERS):
    sca = ax.plot(X[d, :, 0], X[d, :, 1],  # color='k',
                  linewidth=1.,
                  transform=ccrs.Geodetic())
ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='k',
               facecolor=cartopy.feature.COLORS['land_alt1'])
# ax.scatter(X0[:, 0], X0[:, 1])
ax.coastlines()
# plt.colorbar(sca)
plt.tight_layout()
# plt.show()

plt.savefig(figures_dir + "1000simulated_drifters_180days.pdf", format='pdf')
plt.savefig(figures_dir + "1000simulated_drifters_180days.png", format='png')
