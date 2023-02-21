import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy
plt.ioff()
plt.style.use('./misc/paper.mplstyle')

A = np.array([[-50., -20.], [30., 50.]])
B = np.array([[145., 175.], [20., 40.]])
C = np.array([[-130., -100.], [-10., 10.]])

regions = [A, B, C]
names = [r'$A$', r'$B$', r'$C$']
colours = ['#1b9e77', '#d95f02', '#7570b3']

plt.figure(figsize=(6, 3))
proj = cartopy.crs.Robinson()
proj._threshold /= 100.
ax = plt.axes(projection=proj)
ax.set_global()
ax.gridlines(zorder=-10)

for i in range(len(regions)):
    region = regions[i]

    ax.add_patch(mpl.patches.Rectangle((region[0, 0], region[1, 0]),
                 region[0, 1] - region[0, 0],
                 region[1, 1] - region[1, 0],
                 transform=cartopy.crs.PlateCarree(),
                 color=colours[i],
                 linewidth=None,
                 alpha=0.9,
                 ec=None
                 ))

    ax.text(region[0, 0] + 5., region[1, 0] + 5., names[i],
            transform=cartopy.crs.PlateCarree())


ax.add_feature(cartopy.feature.NaturalEarthFeature(
    "physical", "land", "50m"),
               facecolor='k', edgecolor=None, zorder=-11)
plt.tight_layout()

plt.savefig("figures/regions.png")
plt.close()
