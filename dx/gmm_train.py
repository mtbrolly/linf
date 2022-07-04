"""
Gridded Gaussian model.
"""
import numpy as np
import pickle
from grids import GriddedGaussianModel


# Create model
grid_res = 5.  # Grid resolution in degrees.
m = GriddedGaussianModel(n_x=int(360. / grid_res), n_y=int(180. / grid_res))


# Load data.
model_dir = "models/eddie0205_bs8192_lr5em4/"
datas = []
datas_str = ["X", "XVAL", "Y", "YVAL"]
for i in range(len(datas_str)):
    datas.append(np.load(model_dir + datas_str[i] + '.npy'))
[X, XVAL, Y, YVAL] = datas

# Train model.
m.train_model(X, Y)

# Save model.
with open('models/ggm_5degree_ct3/ggm.pickle', 'wb') as f:
    pickle.dump(m, f, pickle.HIGHEST_PROTOCOL)
