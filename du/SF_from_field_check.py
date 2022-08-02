import numpy as np

data_dir = "./data/du/du_from_field/"
rs = np.load(data_dir + "r_range.npy")

r_train = np.load("./data/du/r_train.npy")
du_train = np.load("./data/du/du_train.npy")
# dul == dut  # !!!

ind = r_train == rs[8]
du_ind = du_train[ind[:, 0], :]
