import numpy as np
from scipy.io import savemat

u_all = []
v_all = []

for i in range(1, 101):
    u_all.append(np.load(f"/home/s1511699/sf_ml_snapshots/u_{i:.2f}.npy"))
    v_all.append(np.load(f"/home/s1511699/sf_ml_snapshots/v_{i:.2f}.npy"))

u_all = np.array(u_all)
v_all = np.array(v_all)
np.save("data/du/u_all.npy", u_all)
np.save("data/du/v_all.npy", v_all)
mdic = {'u_all': u_all, 'v_all': v_all}
# savemat("uv.mat", mdic)

u_some = []
v_some = []

for i in range(1, 11):
    u_some.append(np.load(f"/home/s1511699/sf_ml_snapshots/u_{i:.2f}.npy"))
    v_some.append(np.load(f"/home/s1511699/sf_ml_snapshots/v_{i:.2f}.npy"))

u_some = np.array(u_some)
v_some = np.array(v_some)
np.save("data/du/u_some.npy", u_some)
np.save("data/du/v_some.npy", v_some)
mdic = {'u_some': u_some, 'v_some': v_some}
savemat("uv.mat", mdic)
