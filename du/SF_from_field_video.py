import numpy as np
import time
import matplotlib.pyplot as plt
plt.style.use('~/git/linf/figures/experiments.mplstyle')


def sf_direct_estimation(x, y, u_all, v_all, periodic=True):
    """
    Estimate structure functions from a snapshot of gridded velocities.
    -----------------------------
    Assumes a square domain with uniformly spaced gridpoints.

    x, y : float
        1-D arrays of x and y coordinates (not meshgrid arrays)
    u, v : Numpy array
        Gridded 2-D velocity arrays.
    periodic : Boolean
        Whether domain is periodic.
    """

    if periodic:
        r = x[1: int(len(u_all[0, ...]) / 2)] - x[0]
    else:
        r = x[1:] - x[0]

    S = u_all.shape[0]
    SF2l = np.zeros((np.size(r), S))
    SF2t = np.zeros((np.size(r), S))
    SF3l = np.zeros((np.size(r), S))
    SF3t = np.zeros((np.size(r), S))

    for i in range(len(r)):
        print(f'i = {i}')
        t0 = time.time()
        for s in range(u_all.shape[0]):
            u = u_all[s, ...]
            v = v_all[s, ...]
            # Using horizontal r.
            dul0 = u - np.roll(u, (i + 1), axis=0)
            dut0 = v - np.roll(v, (i + 1), axis=0)
            # # Using vertical r.
            dul1 = v - np.roll(v, (i + 1), axis=1)
            dut1 = u - np.roll(u, (i + 1), axis=1)
            # # Collecting these together.
            dul = np.concatenate((dul0, dul1)).flatten()
            dut = np.concatenate((dut0, dut1)).flatten()
            # Add up means for each snapshot.
            SF2l[i, s] += np.mean(dul ** 2)
            SF2t[i, s] += np.mean(dut ** 2)
            SF3l[i, s] += np.mean(dul ** 3)
            SF3t[i, s] += np.mean(dul * dut ** 2)
        T = time.time() - t0
        print(f'Time taken: {T:.2f} seconds.')

    return r, SF2l, SF2t, SF3l, SF3t


# Load a velocity snapshot.
x = np.arange(0.5, 1024, 1.) / 1024 * (2 * np.pi)
y = np.arange(0.5, 1024, 1.) / 1024 * (2 * np.pi)
u_all = np.load(
    "/home/s1511699/git/linf/data/du/u_all.npy")[::5, ...]
v_all = np.load(
    "/home/s1511699/git/linf/data/du/v_all.npy")[::5, ...]
u_all = np.transpose(u_all, (0, 2, 1))
v_all = np.transpose(v_all, (0, 2, 1))

start = time.time()
r, SF2l, SF2t, SF3l, SF3t = sf_direct_estimation(x, y, u_all, v_all)
end = time.time()
print(f'Time taken: {end - start:.2f} seconds.')


results_dir = (
    "/home/s1511699/git/linf/data/du/SF_from_field_video/")
# np.save(results_dir + "r.npy", r)
# np.save(results_dir + "SF2l.npy", SF2l)
# np.save(results_dir + "SF2t.npy", SF2t)
# np.save(results_dir + "SF3l.npy", SF3l)
# np.save(results_dir + "SF3t.npy", SF3t)

r = np.load(results_dir + "r.npy")
SF2l = np.load(results_dir + "SF2l.npy")
SF2t = np.load(results_dir + "SF2t.npy")
SF3l = np.load(results_dir + "SF3l.npy")
SF3t = np.load(results_dir + "SF3t.npy")


# Plot SF3 for different snapshots
colors = ['r', 'g', 'b']
plt.figure()
for s in range(SF2l.shape[1]):
    plt.loglog(r, (SF3l + SF3t)[:, s])
    plt.loglog(r, -(SF3l + SF3t)[:, s], '--')
    # plt.loglog(r, (SF2l + SF2t)[:, s])
    # plt.loglog(r, -(SF2l + SF2t)[:, s], '--')
# plt.plot(np.array([0.01, 0.1]), 1. * np.array([0.01, 0.1]) ** 3, 'g--',
#          label=r'$r^{3}$')
# plt.plot(np.array([0.12, 0.3]), .01 * np.array([0.12, 0.3]) ** 1, 'b--',
#          label=r'$r^{1}$')
# plt.vlines(2 * np.pi / 64, 1e-5, 1e-2, 'grey', '-.',
#            label=r'$l_f$')
# plt.vlines(2 * np.pi / 6, 1e-5, 1e-2, 'grey', '--',
#            label=r'$l_{lsf}$')
# plt.vlines(2 * np.pi / 350, 1e-5, 1e-2, 'grey', ':',
#            label=r'$l_{d}$')
# plt.ylim(1e-7, None)
plt.xlabel(r'$r$')
plt.show()

plt.savefig("/home/s1511699/git/linf/data/du/SF3_video.png", dpi=288)


plt.figure()
for i in tuple([1, 2, 4, 8, 16]):
    plt.plot(SF3l[i, :] / np.mean(SF3l[i, :]), label=f"r = {r[i]:.3f}")
plt.xlabel(r"$s$")
plt.ylabel(r"$S_L(s) / \langle S_L(s) \rangle_s$")
plt.grid()
plt.legend()
plt.show()

CV_SF3l = np.std(SF3l, axis=1) / np.abs(np.mean(SF3l, axis=1))
plt.figure()
plt.loglog(r, CV_SF3l, 'k')
plt.xlabel(r"$r$")
plt.ylabel(r"$CV(S_L(r))$")
plt.grid()
plt.show()


# Get vorticity field.
import pyfftw.interfaces.numpy_fft as fftw  # noqa
uk_all = fftw.rfft2(u_all)
vk_all = fftw.rfft2(v_all)


nx = 1024
ny = nx
nl = ny
nk = int(nx / 2 + 1)
ll = np.append(np.arange(0., nx / 2),
               np.arange(-nx / 2, 0.))
kk = np.arange(0., nk)

k, l = np.meshgrid(kk, ll)  # noqa: E741
ik = 1j * k
il = 1j * l

v_x_all = fftw.irfft2(ik * vk_all)
u_y_all = fftw.irfft2(il * uk_all)
z_all = v_x_all - u_y_all


# Plot u where SF3 is negative at small r.
plt.figure()
plt.pcolormesh(x, y, u_all[1, ...], cmap='RdBu')
# plt.pcolormesh(x, y, z_all[11, ...], cmap='RdBu')
plt.show()


# Plot mean over snapshots of SF3
plt.figure()
SF3lm = SF3l.mean(axis=1)
SF3tm = SF3t.mean(axis=1)
plt.loglog(r, (SF3lm + SF3tm), 'k')
plt.loglog(r, -(SF3lm + SF3tm), 'k--')
plt.plot(np.array([0.01, 0.1]), 1. * np.array([0.01, 0.1]) ** 3, 'g--',
         label=r'$r^{3}$')
plt.plot(np.array([0.12, 0.3]), .01 * np.array([0.12, 0.3]) ** 1, 'b--',
         label=r'$r^{1}$')
plt.vlines(2 * np.pi / 64, 1e-5, 1e-2, 'grey', '-.',
           label=r'$l_f$')
plt.vlines(2 * np.pi / 6, 1e-5, 1e-2, 'grey', '--',
           label=r'$l_{lsf}$')
plt.vlines(2 * np.pi / 350, 1e-5, 1e-2, 'grey', ':',
           label=r'$l_{d}$')
plt.ylim(1e-7, None)
plt.xlabel(r'$r$')
plt.grid()
plt.show()
