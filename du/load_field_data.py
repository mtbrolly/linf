import numpy as np

u_all = []
v_all = []

for i in range(10, 931, 10):
    u_all.append(np.load(f"/home/s1511699/sf_ml_snapshots/u_{i:.2f}.npy"))
    v_all.append(np.load(f"/home/s1511699/sf_ml_snapshots/v_{i:.2f}.npy"))

u_all = np.array(u_all)
v_all = np.array(v_all)
np.save("data/du/u_all.npy", u_all)
np.save("data/du/v_all.npy", v_all)
