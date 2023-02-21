import numpy as np
import matplotlib.pyplot as plt
import cartopy
plt.style.use('./misc/paper.mplstyle')
plt.ioff()

N_C = 32
DT = 4

MODEL_DIR = (f"dx/models/GDP_{DT:.0f}day_NC{N_C}"
             + "_ml_flipout_Adam_tanh_lr5em5_pat50_val20/")

X = np.load(MODEL_DIR + "homogeneous_release_10year_rejection.npy")

# plt.figure()
# plt.scatter(X[:, -1, 0], X[:, -1, 1], color='k')
# plt.show()

# t = 10 * 365
# tn = int(t / DT)

for tn in range(X.shape[1]):
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=cartopy.crs.Robinson(central_longitude=0.))
    ax.scatter(X[:, tn, 0], X[:, tn, 1], s=30, color='g', marker='.',
               transform=cartopy.crs.PlateCarree(), alpha=0.2)
    ax.add_feature(cartopy.feature.NaturalEarthFeature(
        "physical", "land", "50m"),
        facecolor='k', edgecolor=None, zorder=100)
    plt.title(f'$t =$ {(tn * DT) // 365:.0f} years, {(tn * DT) % 365} days')
    plt.tight_layout()
    plt.savefig(
        MODEL_DIR
        + "figures/homogeneous_release_10year_rejection/"
        + "t{:.0f}.png".format(tn),
        dpi=150)
    # plt.show()
    plt.close()

# cd ~/git/linf/dx/models/GDP_4day_NC32_ml_flipout_Adam_tanh_lr5em5_pat50_val20/figures/homogeneous_release_10year_rejection
# ffmpeg -framerate 10 -i t%d.png -pix_fmt yuv420p tracer_alpha.mp4
