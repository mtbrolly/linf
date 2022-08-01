import numpy as np

data_dir = "/home/s1511699/git/linf/data/du/du_from_field/"
rs = np.load(data_dir + "r_range.npy")

dul = np.load(data_dir + f"dul_{0:.0f}.npy")
rn = [dul.size, ]

for i in range(1, 17):
    dul_i = np.load(data_dir + f"dul_{i:.0f}.npy")
    dul = np.concatenate((dul, dul_i), axis=0)
    rn.append(dul_i.size)
    del dul_i

np.save(data_dir + "dul.npy", dul)
del dul

r = np.ones((rn[0], )) * rs[0]
for i in range(1, 17):
    r = np.concatenate((r, np.ones((rn[i], )) * rs[i]))

np.save(data_dir + "r.npy", r)
del r

dut = np.load(data_dir + f"dul_{0:.0f}.npy")

for i in range(1, 17):
    dut = np.concatenate((dut, np.load(data_dir + f"dut_{i:.0f}.npy")), axis=0)

np.save(data_dir + "dut.npy", dut)
del dut

dul = np.load(data_dir + "dul.npy")
dut = np.load(data_dir + "dut.npy")
du = np.concatenate((dul[:, None], dut[:, None]), axis=1)
del dul, dut
np.save(data_dir + "du.npy", du)


# Shuffling and partitioning
r = np.load(data_dir + "r.npy")[:, None]
rng = np.random.default_rng(seed=1)
rng.shuffle(r)

N_train = r.size // 2
np.save(data_dir + "r_train.npy", r[:N_train, :])
np.save(data_dir + "r_test.npy", r[N_train:, :])
del r

du = np.load(data_dir + "du.npy")
rng = np.random.default_rng(seed=1)
rng.shuffle(du)

np.save(data_dir + "du_train.npy", du[:N_train, :])
np.save(data_dir + "du_test.npy", du[N_train:, :])
del du
