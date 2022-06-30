import numpy as np
import matplotlib.pyplot as plt
import time
plt.style.use('~/github/linf/figures/experiments.mplstyle')


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

    SF2l = np.zeros(np.shape(r))
    SF2t = np.zeros(np.shape(r))
    SF3l = np.zeros(np.shape(r))
    SF3t = np.zeros(np.shape(r))

    for i in range(len(r)):
        print(f'i = {i}')
        t0 = time.time()
        for s in range(u_all.shape[0]):
            u = u_all[s, ...]
            v = v_all[s, ...]
            # Using horizontal r.
            dul0 = u - np.roll(u, (i + 1), axis=0)
            dut0 = v - np.roll(v, (i + 1), axis=0)
            # Using vertical r.
            dul1 = v - np.roll(v, (i + 1), axis=1)
            dut1 = u - np.roll(u, (i + 1), axis=1)
            # Collecting these together.
            dul = np.concatenate((dul0, dul1))
            dut = np.concatenate((dut0, dut1))
            # Add up means for each snapshot.
            SF2l[i] += np.mean(dul ** 2)
            SF2t[i] += np.mean(dut ** 2)
            SF3l[i] += np.mean(dul ** 3)
            SF3t[i] += np.mean(dul * dut ** 2)
        T = time.time() - t0
        print(f'Time taken: {T:.1f} seconds.')
    # Divide to average over snapshots.
    SF2l /= u_all.shape[0]
    SF2t /= u_all.shape[0]
    SF3l /= u_all.shape[0]
    SF3t /= u_all.shape[0]

    return r, SF2l, SF2t, SF3l, SF3t


# Load a velocity snapshot.
x = np.arange(0.5, 1024, 1.) / 1024 * (2 * np.pi)
y = np.arange(0.5, 1024, 1.) / 1024 * (2 * np.pi)
# u_all = np.load(
#     "/home/s1511699/github/linf/data/sf_ml_snapshots/u_all.npy")
# v_all = np.load(
#     "/home/s1511699/github/linf/data/sf_ml_snapshots/v_all.npy")
u_all = np.load("./u_all.npy")
v_all = np.load("./v_all.npy")

r, SF2l, SF2t, SF3l, SF3t = sf_direct_estimation(x, y, u_all, v_all)

np.save("./r.npy", r)
np.save("./SF2l.npy", SF2l)
np.save("./SF2t.npy", SF2t)
np.save("./SF3l.npy", SF3l)
np.save("./SF3t.npy", SF3t)

# Check correspondence between SF2l and SF2t.
SF2t_rc = np.gradient(r * SF2l, r[1] - r[0])

plt.figure()
plt.plot(r, SF2t, 'k')
plt.plot(r, SF2t_rc, 'b--')
plt.show()


# Check correspondence between SF3l and SF3t.
dSF3l_dr = (SF3l[1:] - SF3l[:-1]) / (r[1] - r[0])
dSF3l_dr_g = np.gradient(SF3l, r[1] - r[0])

plt.figure()
plt.plot(r, SF3t, 'k')
# plt.plot(r[:-1], r[:-1] / 3 * dSF3l_dr, 'r--')
plt.plot(r, r / 3 * dSF3l_dr_g, 'b--')
plt.show()

plt.figure()
plt.loglog(r, np.abs(SF3t), 'k')
# plt.plot(r[:-1], r[:-1] / 3 * dSF3l_dr, 'r--')
plt.loglog(r, np.abs(r / 3 * dSF3l_dr_g), 'b--')
plt.show()


# Plot third-order structure functions
plt.figure()
plt.semilogx(r, SF3l / r + SF3t / r, 'k', label=r'$S_L(r) + S_T(r)$')
plt.semilogx(r, SF3l / r, 'b', label=r'$S_L(r)$')
plt.semilogx(r, SF3t / r, 'g', label=r'$S_T(r)$')
plt.xlabel(r'$r$')
plt.legend()
plt.grid()
plt.show()


plt.figure()
plt.loglog(r, np.abs(SF3l), 'k*-')
plt.plot(np.array([0.01, 0.05]), 100 * np.array([0.01, 0.05]) ** 3, 'r--',
         label=r'$r^{3}$')
plt.plot(np.array([0.12, 0.3]), .1 * np.array([0.12, 0.3]) ** 1, 'b--',
         label=r'$r^{1}$')
plt.vlines(2 * np.pi / 64, 1e-3, 0.1, 'darkgrey', '-.',
           label=r'$l_f$')
plt.vlines(2 * np.pi / 6, 1e-3, 0.1, 'grey', '-.',
           label=r'$l_{lsf}$')
plt.ylabel(r'$|S_L(r)|$')
plt.xlabel(r'$r$')
plt.legend()
plt.grid()
plt.show()


# Plot second-order structure functions
plt.figure()
plt.loglog(r, SF2l + SF2t, 'k-', label=r'$S_2^{(L)}+S_2^{(T)}(r)$')
plt.loglog(r, SF2l, 'b-', label=r'$S_2^{(L)}(r)$')
plt.loglog(r, SF2t, 'g-', label=r'$S_2^{(T)}(r)$')

plt.vlines(2 * np.pi / 64, 4e-2, 0.5, 'grey', '-.',
           label=r'$l_f$')
plt.vlines(2 * np.pi / 6, 4e-2, 0.5, 'grey', '--',
           label=r'$l_{lsf}$')
plt.vlines(2 * np.pi / 350, 4e-2, 0.5, 'grey', ':',
           label=r'$l_{d}$')
plt.plot([0.01, 0.08], 124 * np.array([0.01, 0.08]) ** 2., 'r--',
         label=r'$r^{2}$')
plt.plot([0.11, 0.25], np.array([0.11, 0.25]) ** (1.), 'm--',
         label=r'$r^{1}$')
plt.grid(True)
plt.xlabel(r'$r$')
plt.title(r'Longitudinal and transverse second-order structure function')
plt.legend()
plt.tight_layout()
plt.show()
