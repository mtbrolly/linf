"""
Prepare velocity difference data for training.
"""

# import dask.array as da
import numpy as np
import time
import matplotlib.pyplot as plt
from numba import jit
plt.style.use('~/git/linf/figures/experiments.mplstyle')

x, y = np.meshgrid(
    np.arange(0.5, 1024, 1.) / 1024 * (2 * np.pi),
    np.arange(0.5, 1024, 1.) / 1024 * (2 * np.pi))

u = np.load("data/du/u_all.npy")[:1, ...]
v = np.load("data/du/v_all.npy")[:1, ...]

u = np.transpose(u, (0, 2, 1))
v = np.transpose(v, (0, 2, 1))


# Create a random sample of velocity differences.
rng = np.random.default_rng(seed=10)
N = 50000000
s = rng.integers(u.shape[0], size=N)
x1 = rng.integers(1024, size=N)
y1 = rng.integers(1024, size=N)
x2 = np.mod(
    x1 + np.ceil(rng.integers(1024, size=N)).astype(int),
    1024)
y2 = y1  # Only sample horizontal separation vectors.

dx = x[x2, y2] - x[x1, y1]
dy = y[x2, y2] - y[x1, y1]
dX = np.concatenate((dx[:, None], dy[:, None]), axis=1)
del dx, dy
du = u[s, x2, y2] - u[s, x1, y1]
dv = v[s, x2, y2] - v[s, x1, y1]
dU = np.concatenate((du[:, None], dv[:, None]), axis=1)
del du, dv


# Wrap dX.
dX[:, 0] -= 2 * np.pi * (dX[:, 0] > np.pi)
dX[:, 0] += 2 * np.pi * (dX[:, 0] < -np.pi)
dX[:, 1] -= 2 * np.pi * (dX[:, 1] > np.pi)
dX[:, 1] += 2 * np.pi * (dX[:, 1] < -np.pi)
r = np.sqrt(dX[:, 0] ** 2 + dX[:, 1] ** 2)
ry = dX[:, 1]

# Remove any separation vectors with magnitude 0.
rnz_ind = dX[:, 1] != 0
r = r[rnz_ind]
dX = dX[rnz_ind, :]
dU = dU[rnz_ind, :]

# Calculate longitudinal and transverse velocity differences.
# delta_lt = np.zeros_like(dU)
# delta_lt[:, 0] = np.sum(dU * dX, axis=1) / r
# delta_lt[:, 1] = (dU[:, 1] * dX[:, 0] - dU[:, 0] * dX[:, 1]) / r
delta_lt2 = np.zeros_like(dU)
delta_lt2[:, 0] = dU[:, 1]
delta_lt2[:, 1] = dU[:, 0]
delta_lt = delta_lt2
del dU

# =============================================================================
# N_train = r.size // 2
# r = r.reshape((r.size, 1))
# np.save("data/du/r_train.npy",
#         r[:N_train, :])
# np.save("data/du/r_test.npy",
#         [N_train:, :])
# np.save("data/du/du_train.npy",
#         delta_lt[:N_train, :])
# np.save("data/du/du_test.npy",
#         delta_lt[N_train:, :])
# =============================================================================

# Binning increments by r.
rm = 2 * np.pi * np.arange(1, 513) / 1024
rs = np.zeros((rm.size + 1,))
rs[0] = 0.5 * rm[0]
rs[1:] = rm + 0.5 * rm[0]

# delta_lt = du_all
# r = r_all
# r = r[:, 0]

binned_r = np.digitize(r, rs)


@jit(nopython=True, parallel=True)  # noqa
def p_for(binned_r, delta_lt):
    mean_l = np.zeros_like(rm)
    mean_t = np.zeros_like(rm)
    S2l = np.zeros_like(rm)
    S2t = np.zeros_like(rm)
    S3l = np.zeros_like(rm)
    S3t = np.zeros_like(rm)
    S2N = np.zeros_like(rm)

    for ri in range(rm.size):
        print(ri)
        sample = delta_lt[binned_r == ri + 1, :]
        S2l[ri] = np.var(sample[:, 0])
        S2t[ri] = np.var(sample[:, 1])
        S3l[ri] = np.mean(sample[:, 0] ** 3)
        S3t[ri] = np.mean(sample[:, 0] * (sample[:, 1] ** 2))
        S2N[ri] = sample.size
        mean_l[ri] = np.mean(sample[:, 0])
        mean_t[ri] = np.mean(sample[:, 1])
    return mean_l, mean_t, S2l, S2t, S3l, S3t, S2N


t0 = time.time()
mean_l, mean_t, S2l, S2t, S3l, S3t, S2N = p_for(binned_r, delta_lt)
t1 = time.time()
print(f'Took {(t1-t0) / 512:.3f} seconds per r.')

# np.save("/home/s1511699/git/linf/data/du/real_binned_estimates/r.npy", r)
# np.save("/home/s1511699/git/linf/data/du/real_binned_estimates/S2.npy",
#         S2l + S2t)
# np.save("/home/s1511699/git/linf/data/du/real_binned_estimates/S2l.npy", S2l)
# np.save("/home/s1511699/git/linf/data/du/real_binned_estimates/S2t.npy", S2t)

# results_dir = (
#     "/home/s1511699/git/linf/data/du/SF_direct_1/")
# np.save(results_dir + "r.npy", rm)
# np.save(results_dir + "S2l.npy", S2l)
# np.save(results_dir + "S2t.npy", S2t)
# np.save(results_dir + "S3l.npy", S3l)
# np.save(results_dir + "S3t.npy", S3t)


fig_dir = "du/figures/real_binned_estimates/"

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
# plt.savefig(fig_dir + "S2.png", format='png', dpi=576)

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
# plt.savefig(fig_dir + "S2lt.png", format='png', dpi=576)

plt.figure()
plt.semilogx(rm, S3l + S3t, 'k-')
plt.vlines(2 * np.pi / 64, -0.2, 0.2, 'grey', '-.',
           label=r'$l_f$')
plt.vlines(2 * np.pi / 6, -0.2, 0.2, 'grey', '--',
           label=r'$l_{lsf}$')
plt.vlines(2 * np.pi / 350, -0.2, 0.2, 'grey', ':',
           label=r'$l_{d}$')
plt.grid(True)
plt.xlabel(r'$r$')
plt.ylabel(r'$S_3(r)$')
plt.title(r'Isotropic third-order structure function')
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig(fig_dir + "S3_o_r.png", format='png', dpi=576)

plt.figure()
plt.loglog(rm, np.abs(S3l + S3t), 'k-')
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
# plt.savefig(fig_dir + "S3_abs.png", format='png', dpi=576)
