import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
# import cmocean
from drifter_data import process_raw_df
plt.style.use('./figures/posters.mplstyle')
plt.ioff()

df = process_raw_df()
ids = np.unique(df.index.get_level_values(level=0))

df = df[:]
dur = df.groupby(level=0).size()

ids180 = df.index.get_level_values(0).unique()[dur > 180 * 4]
desired_dur = 180 * 4

np.random.seed(1)
Np = 1000
chosen_ids = np.random.choice(ids180, size=Np, replace=False)

X0 = np.zeros((Np, 2))
for i in range(Np):
    X0[i, :] = df.xs(chosen_ids[i], level=0)[0:1].values

X0 = X0[:, ::-1]
np.save("data/sampled_drifters_inits.npy", X0)


plt.figure(figsize=(15, 7.5))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0.))
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_extent([-180., 180., -90., 90.], crs=ccrs.PlateCarree())
plt.title("Drifter trajectories")
for i in range(Np):  # len(ids)):
    # d = ids[i]
    # d = int(np.floor(len(ids) * np.random.rand(1)))
    d_data = df.xs(chosen_ids[i], level=0)[:desired_dur]
    x = d_data.lon.to_numpy()
    y = d_data.lat.to_numpy()
    sca = ax.plot(x, y,  # color='k',
                  linewidth=1.,
                  transform=ccrs.Geodetic())
ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='k',
               facecolor=cartopy.feature.COLORS['land_alt1'])
ax.coastlines()
# plt.colorbar(sca)
plt.tight_layout()
# plt.show()

plt.savefig("figures/1000drifters_without_gaps_same_length.png", format='png')
