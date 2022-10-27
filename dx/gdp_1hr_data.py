"""
Accessed at https://www.nodc.noaa.gov/archive/arc0199/0248584/1.1/data/0-data/
at 12:15 GMT on 04/10/2022.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
plt.ioff()

ds = xr.open_dataset("data/GDP/1hr/gdp_v2.00.nc", chunks={})

traj_idx = np.insert(np.cumsum(ds.rowsize.values), 0, 0)

# Check for gaps

i = 0
gapped = 0
for i in range(1000):  # len(traj_idx)):
    i = 5270 - i
    t = ds.time[slice(traj_idx[i], traj_idx[i + 1])].compute().data
    dt = t[1:] - t[:-1]
    n_gaps = np.sum(dt != np.timedelta64(3600000000000))
    if n_gaps > 0:
        gapped += 1
        print(f"Found {n_gaps:.0f} gaps! :(")

plt.figure()
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
ax.set_global()
for i in range(100):
    print(i)
    x = ds.longitude[slice(traj_idx[i], traj_idx[i + 1])].compute().data
    y = ds.latitude[slice(traj_idx[i], traj_idx[i + 1])].compute().data
    sca = ax.plot(x, y,
                  linewidth=0.5,
                  transform=ccrs.Geodetic())
ax.add_feature(cartopy.feature.LAND, zorder=100,
               facecolor='k')
plt.tight_layout()
plt.show()
