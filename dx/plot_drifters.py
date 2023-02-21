import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
# import cmocean
from dx.drifter_data import process_raw_df
plt.style.use('./misc/slides.mplstyle')
plt.ioff()

df = process_raw_df()
ids = np.unique(df.index.get_level_values(level=0))

df = df[:]
# dur = df.groupby(level=0).size()

# ids180 = df.index.get_level_values(0).unique()[dur > 180 * 4]
# desired_dur = 180 * 4

# np.random.seed(1)
Np = 1000
# chosen_ids = np.random.choice(ids180, size=Np, replace=False)

# X0 = np.zeros((Np, 2))
# for i in range(Np):
#     X0[i, :] = df.xs(chosen_ids[i], level=0)[0:1].values

# X0 = X0[:, ::-1]
# np.save("data/GDP/coords/sampled_drifters_inits.npy", X0)


plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
# ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_extent([-40., 5., 30., 60.], crs=ccrs.PlateCarree())
ax.set_global()
# plt.title("Drifter trajectories")
for i in range(Np):
    print(i)
    # d = ids[i]
    # d = int(np.floor(len(ids) * np.random.rand(1)))
    d_data = df.xs(ids[i], level=0)  # .xs(chosen_ids[i], level=0)[:desired_dur]
    x = d_data.lon.to_numpy()  # [::4]  # !!!
    y = d_data.lat.to_numpy()  # [::4]
    sca = ax.plot(x, y,
                  linewidth=1.25,
                  transform=ccrs.Geodetic())
ax.add_feature(cartopy.feature.NaturalEarthFeature("physical", "land", "50m"),
               facecolor='k', edgecolor=None, zorder=100)
plt.tight_layout()
# plt.show()

# plt.savefig("figures/1000drifters_without_gaps_same_length_13_09_22_2days.png",
#             format='png')
# plt.savefig("figures/drifter_trajectories_sample_Leeds_lowres.png", dpi=150)
plt.savefig("figures/drifter_trajectories_sample_Leeds_1000.pdf")
