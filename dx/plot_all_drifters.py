import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
# import cmocean
from dx.drifter_data import process_raw_df
plt.style.use('./misc/poster.mplstyle')
plt.ioff()

df = process_raw_df()
ids = np.unique(df.index.get_level_values(level=0))

ids = ids[::100]  # !!!

# df = df[:]
# dur = df.groupby(level=0).size()

# ids180 = df.index.get_level_values(0).unique()[dur > 180 * 4]

# np.random.seed(1)
# Np = 1000
# chosen_ids = np.random.choice(ids180, size=Np, replace=False)

Np = len(ids)

X0 = np.zeros((Np, 2))
for i in range(Np):
    # X0[i, :] = df.xs(chosen_ids[i], level=0)[0:1].values
    X0[i, :] = df.xs(ids[i], level=0)[0:1].values

X0 = X0[:, ::-1]
# np.save("data/GDP/coords/sampled_drifters_inits.npy", X0)


plt.figure(figsize=(8.5085, 8.5085 / 2))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
# ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# ax.set_extent([-360., 180., -90., 90.], crs=ccrs.PlateCarree())
ax.set_global()
# plt.title("Drifter trajectories")
for i in range(Np):
    print(i)
    # d = ids[i]
    # d = int(np.floor(len(ids) * np.random.rand(1)))
    # d_data = df.xs(chosen_ids[i], level=0)
    d_data = df.xs(ids[i], level=0)
    x = d_data.lon.to_numpy()
    y = d_data.lat.to_numpy()
    sca = ax.plot(x, y,
                  linewidth=0.5,
                  transform=ccrs.Geodetic())
ax.add_feature(cartopy.feature.LAND, zorder=100,
               facecolor='k')
# ax.coastlines()
plt.tight_layout()
# plt.show()

plt.savefig("figures/all_drifters.png",
            format='png')
