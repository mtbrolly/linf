"""
Prepare velocity difference data for training.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('~/github/linf/figures/experiments.mplstyle')

x, y = np.meshgrid(
    np.arange(0.5, 1024, 1.) / 1024 * (2 * np.pi),
    np.arange(0.5, 1024, 1.) / 1024 * (2 * np.pi))

# =============================================================================
# u = np.zeros((100, 1024, 1024))
# v = np.zeros((100, 1024, 1024))
# for i in range(100):
#     t = f"{(i + 1) * 0.1 :.2f}"
#     print(t)
#     u[i, ...] = np.load("/home/s1511699/github/linf/data/sf_ml_snapshots/u_"
#                         + t + ".npy")
#     v[i, ...] = np.load("/home/s1511699/github/linf/data/sf_ml_snapshots/v_"
#                         + t + ".npy")
#
# np.save("/home/s1511699/github/linf/data/sf_ml_snapshots/u_all.npy", u)
# np.save("/home/s1511699/github/linf/data/sf_ml_snapshots/v_all.npy", v)
# =============================================================================

u = np.load("/home/s1511699/github/linf/data/sf_ml_snapshots/u_all.npy")
v = np.load("/home/s1511699/github/linf/data/sf_ml_snapshots/v_all.npy")

# =============================================================================
# plt.figure()
# plt.pcolormesh(x, y, u[0, ...].T, shading='gouraud', cmap='RdBu')
# plt.show()
# =============================================================================


# Create a random sample of velocity differences.
dU = []
dX = []

rng = np.random.default_rng(seed=10)
N = 50000000
s = rng.integers(u.shape[0], size=N)
x1 = rng.integers(1024, size=N)
# x2 = np.mod(x1 + rng.integers(256, size=N), 1024).astype(int)
x2 = np.mod(
    x1 + np.ceil(rng.integers(1024, size=N)).astype(int),
    1024)
y1 = rng.integers(1024, size=N)
# y2 = np.mod(y1 + rng.integers(256, size=N), 1024).astype(int)
y2 = y1

for n in range(N):
    dU.append([u[s[n], x1[n], y1[n]] - u[s[n], x2[n], y2[n]],
               v[s[n], x1[n], y1[n]] - v[s[n], x2[n], y2[n]]])
    dX.append([x[x1[n], y1[n]] - x[x2[n], y2[n]],
               y[x1[n], y1[n]] - y[x2[n], y2[n]]])

dU = np.array(dU)
dX = np.array(dX)

# Wrap dX.
dX[:, 0] -= 2 * np.pi * (dX[:, 0] > np.pi)
dX[:, 0] += 2 * np.pi * (dX[:, 0] < -np.pi)
dX[:, 1] -= 2 * np.pi * (dX[:, 1] > np.pi)
dX[:, 1] += 2 * np.pi * (dX[:, 1] < -np.pi)
r = np.sqrt(dX[:, 0] ** 2 + dX[:, 1] ** 2)

rnz_ind = dX[:, 1] != 0
r = r[rnz_ind]
dX = dX[rnz_ind, :]
dU = dU[rnz_ind, :]

# Calculate longitudinal and transverse velocity differences.
delta_lt = np.zeros_like(dU)
delta_lt[:, 0] = np.sum(dU * dX, axis=1) / r
delta_lt[:, 1] = (dU[:, 1] * dX[:, 0] - dU[:, 0] * dX[:, 1]) / r

# r = r[np.where(r != 0)]
# delta_lt = delta_lt[np.where(r != 0)[0], :]

# np.save("/home/s1511699/github/linf/data/sf_ml_snapshots/r.npy", r)
# np.save("/home/s1511699/github/linf/data/sf_ml_snapshots/dul.npy",
#         delta_lt[:, 0])
# np.save("/home/s1511699/github/linf/data/sf_ml_snapshots/dut.npy",
#         delta_lt[:, 1])


# Binning increments by r.
# rs = np.logspace(-2, -0.1, 100)
# rs = np.linspace(2. * np.pi / 200, 2. * np.pi / 3, 500)
# rm = (rs[1:] + rs[:-1]) / 2

rm = 2 * np.pi * np.arange(1, 513) / 1024
# rm = rm[::10]
rs = np.zeros((rm.size + 1,))
rs[0] = 0.5 * rm[0]
rs[1:] = rm + 0.5 * rm[0]


mean_l = np.zeros_like(rm)
mean_t = np.zeros_like(rm)
S2l = np.zeros_like(rm)
S2t = np.zeros_like(rm)
S3 = np.zeros_like(rm)
S2N = np.zeros_like(rm)
for ri in range(rm.size):
    sample = delta_lt[r > rs[ri], :][r[r > rs[ri]] < rs[ri + 1], :]
    S2l[ri] = np.var(sample[:, 0])
    S2t[ri] = np.var(sample[:, 1])
    S3[ri] = np.mean(sample[:, 0] ** 3) + np.mean(
        sample[:, 0] * sample[:, 1] ** 2)
    S2N[ri] = sample.size
    mean_l[ri] = np.mean(sample[:, 0])
    mean_t[ri] = np.mean(sample[:, 1])

# np.save("/home/s1511699/github/linf/data/sf_ml_snapshots/rm.npy", rm)
# np.save("/home/s1511699/github/linf/data/sf_ml_snapshots/S2.npy", S2l + S2t)
# np.save("/home/s1511699/github/linf/data/sf_ml_snapshots/S2l.npy", S2l)
# np.save("/home/s1511699/github/linf/data/sf_ml_snapshots/S2t.npy", S2t)


plt.figure()
plt.loglog(rm, S2l + S2t, 'k-')
plt.vlines(2 * np.pi / 64, 0.1, 0.5, 'grey', '-.',
           label=r'$l_f$')
plt.vlines(2 * np.pi / 6, 0.1, 0.5, 'grey', '--',
           label=r'$l_{lsf}$')
plt.vlines(2 * np.pi / 350, 0.1, 0.5, 'grey', ':',
           label=r'$l_{d}$')
plt.plot([0.04, 0.08], 40 * np.array([0.04, 0.08]) ** 2., 'g--',
         label=r'$r^{2}$')
plt.plot([0.11, 0.25], 1.5 * np.array([0.11, 0.25]) ** (2. / 3.), 'b--',
         label=r'$r^{2/3}$')
plt.grid(True)
plt.xlabel(r'$r$')
plt.ylabel(r'$S_2(r)$')
plt.title(r'Isotropic second-order structure function')
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig("./figures/S2.png", format='png', dpi=576)

plt.figure()
plt.loglog(rm, S2l, 'k-', label=r'$S_2^{(L)}(r)$')
plt.loglog(rm, S2t, 'k-.', label=r'$S_2^{(T)}(r)$')
plt.vlines(2 * np.pi / 64, 4e-2, 0.5, 'grey', '-.',
           label=r'$l_f$')
plt.vlines(2 * np.pi / 6, 4e-2, 0.5, 'grey', '--',
           label=r'$l_{lsf}$')
plt.vlines(2 * np.pi / 350, 4e-2, 0.5, 'grey', ':',
           label=r'$l_{d}$')
plt.plot([0.04, 0.08], 16 * np.array([0.04, 0.08]) ** 2., 'g--',
         label=r'$r^{2}$')
plt.plot([0.11, 0.25], np.array([0.11, 0.25]) ** (1.), 'b--',
         label=r'$r^{1}$')
plt.grid(True)
plt.xlabel(r'$r$')
plt.title(r'Longitudinal and transverse second-order structure function')
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig("./figures/S2lt.png", format='png', dpi=576)

plt.figure()
plt.semilogx(rm, S3 / rm, 'k-')
plt.vlines(2 * np.pi / 64, -0.2, 0.2, 'grey', '-.',
           label=r'$l_f$')
plt.vlines(2 * np.pi / 6, -0.2, 0.2, 'grey', '--',
           label=r'$l_{lsf}$')
plt.vlines(2 * np.pi / 350, -0.2, 0.2, 'grey', ':',
           label=r'$l_{d}$')
plt.grid(True)
plt.xlabel(r'$r$')
plt.ylabel(r'$S_3(r)/r$')
plt.title(r'Isotropic third-order structure function')
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig("./figures/S3_o_r.png", format='png', dpi=576)

plt.figure()
plt.loglog(rm, np.abs(S3), 'k-')
plt.vlines(2 * np.pi / 64, -0.2, 0.2, 'grey', '-.',
           label=r'$l_f$')
plt.vlines(2 * np.pi / 6, -0.2, 0.2, 'grey', '--',
           label=r'$l_{lsf}$')
plt.vlines(2 * np.pi / 350, -0.2, 0.2, 'grey', ':',
           label=r'$l_{d}$')
plt.grid(True)
plt.xlabel(r'$r$')
plt.ylabel(r'$|S_3(r)|$')
plt.title(r'Isotropic third-order structure function')
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig("./figures/S3_abs.png", format='png', dpi=576)
