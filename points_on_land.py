"""
Generate points distributed uniformly(ish) at random on land.

Notes:

    Points are sampled uniformly on [-180, 180] x [-90, 90] rectangle, rather
    than uniformly on the sphere, so the poles are sampled more per area than
    equatorial regions.

    The land mask used is 1:110m scale version; higher resolution 1:10m scale
    version takes too long.
"""

import time
import regionmask
import numpy as np
from shapely.geometry import MultiPoint
from shapely.ops import unary_union
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
land_poly = unary_union(land.polygons)


N_POINTS = 1000000  # int(2.8e7)
rng = np.random.default_rng(seed=123)
r1 = rng.uniform(size=N_POINTS)
r2 = rng.uniform(size=N_POINTS)
lonrad = 2. * np.pi * r1
phi = np.arccos(2. * r2 - 1.)
latrad = np.pi / 2. - phi
points = (np.concatenate((lonrad[None, :], latrad[None, :]), axis=0)
          * 360. / (2. * np.pi))
points[0, :] -= 180.


# =============================================================================
# plt.figure()
# ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0.))
# ax.scatter(points[:, 0], points[:, 1], c='k', s=0.01,
#            transform=ccrs.PlateCarree())
# ax.set_global()
# ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# plt.tight_layout()
# plt.show()
# =============================================================================


points_list = [points[:, i] for i in range(points.shape[1])]
geom_points = MultiPoint(points_list)
on_land = []

t0 = time.time()
for i, x in enumerate(geom_points):
    if np.mod(i, 1000) == 0 and i > 0:
        t1 = time.time()
        print("Estimated time until completion: "
              + f"{(t1 - t0) / i * (N_POINTS - i) / 60:.1f} "
              + "minutes")
    on_land.append(x.intersects(land_poly))

# points_on_land = np.array(geom_points)[on_land]
points_on_sea = np.array(geom_points)[~np.array(on_land)]
np.save("data/points_on_sea_1e6.npy", points_on_sea)
# points_on_land = np.load("data/points_on_sea.npy")

# np.save("data/points_on_land.npy", points_on_land)
# points_on_land = np.load("data/points_on_land.npy")


plt.figure()
ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0.))
for geom in land_poly.geoms:
    ax.fill(*geom.exterior.xy, 'k', transform=ccrs.PlateCarree())
ax.scatter(points_on_sea[:, 0], points_on_sea[:, 1], c='b', s=0.01,
           transform=ccrs.PlateCarree())
ax.set_global()
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
plt.tight_layout()
plt.show()
