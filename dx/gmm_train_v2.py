"""
Gridded Gaussian model.
"""
import numpy as np
import pickle
from tools import grids


# Create model
grid_res = 1.  # Grid resolution in degrees.
m = grids.GriddedGaussianModel(
    n_x=int(360. / grid_res), n_y=int(180. / grid_res))


# Load data

DT = 2

DATA_DIR = f"data/GDP/{DT:.0f}day/"

X = np.load(DATA_DIR + "X0_train.npy")
Y = np.load(DATA_DIR + "DX_train.npy")

# =============================================================================
# # Concatenate with test data for map of drifter coverage
#
# XVAL = np.load(DATA_DIR + "X0_test.npy")
# YVAL = np.load(DATA_DIR + "DX_test.npy")
#
# X = np.concatenate((X, XVAL))
# Y = np.concatenate((Y, YVAL))
#
# del XVAL, YVAL
# =============================================================================

# Train model.
m.fit(X, Y)

# Save model.
with open('dx/models/ggm/1deg/ggm.pickle', 'wb') as f:
    pickle.dump(m, f, pickle.HIGHEST_PROTOCOL)
