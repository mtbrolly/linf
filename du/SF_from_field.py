import numpy as np
import time


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

    sample_rate = 10
    ns = int((1024 ** 2 * 2) / sample_rate + 1)
    dul_all = np.zeros((ns, r.size))
    dut_all = np.zeros((ns, r.size))
    r_all = np.zeros((ns, r.size))

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
            # dul = dul1
            # dut = dut1
            rs = np.ones_like(dul[::sample_rate]) * (i + 1) * 2. * np.pi / 1024  # noqa
            # Stockpile these increments.
            dul_all[:, i] = dul[::sample_rate]
            dut_all[:, i] = dut[::sample_rate]
            r_all[:, i] = rs
            # Add up means for each snapshot.
            SF2l[i] += np.mean(dul ** 2)
            SF2t[i] += np.mean(dut ** 2)
            SF3l[i] += np.mean(dul ** 3)
            SF3t[i] += np.mean(dul * dut ** 2)
        T = time.time() - t0
        print(f'Time taken: {T:.2f} seconds.')
    # Divide to average over snapshots.
    SF2l /= u_all.shape[0]
    SF2t /= u_all.shape[0]
    SF3l /= u_all.shape[0]
    SF3t /= u_all.shape[0]

    return r, SF2l, SF2t, SF3l, SF3t, r_all, dul_all, dut_all


# Load a velocity snapshot.
x = np.arange(0.5, 1024, 1.) / 1024 * (2 * np.pi)
y = np.arange(0.5, 1024, 1.) / 1024 * (2 * np.pi)
u_all = np.load(
    "/home/s1511699/git/linf/data/du/u_all.npy")[:1, ...]
v_all = np.load(
    "/home/s1511699/git/linf/data/du/v_all.npy")[:1, ...]
u_all = np.transpose(u_all, (0, 2, 1))
v_all = np.transpose(v_all, (0, 2, 1))


r, SF2l, SF2t, SF3l, SF3t, r_all, dul_all, dut_all = sf_direct_estimation(
    x, y, u_all, v_all)

r_all = r_all.flatten()
dul_all = dul_all.flatten()
dut_all = dut_all.flatten()
du_all = np.concatenate((dul_all[:, None], dut_all[:, None]), axis=1)


# Shuffle data randomly.
shuffler_rng = np.random.default_rng(seed=1)
shuffled_index = shuffler_rng.permutation(np.arange(r_all.size))
r_all = r_all[shuffled_index, ...]
du_all = du_all[shuffled_index, ...]

# N_train = r_all.size // 2
# r_all = r_all.reshape((r_all.size, 1))
# np.save("data/du/r_train.npy",
#         r_all[:N_train, :])
# np.save("data/du/r_test.npy",
#         r_all[N_train:, :])
# np.save("data/du/du_train.npy",
#         du_all[:N_train, :])
# np.save("data/du/du_test.npy",
#         du_all[N_train:, :])


# results_dir = (
#     "/home/s1511699/git/linf/data/du/SF_from_field_1/")
# np.save(results_dir + "r.npy", r)
# np.save(results_dir + "SF2l.npy", SF2l)
# np.save(results_dir + "SF2t.npy", SF2t)
# np.save(results_dir + "SF3l.npy", SF3l)
# np.save(results_dir + "SF3t.npy", SF3t)
