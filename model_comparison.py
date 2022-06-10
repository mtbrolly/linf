"""
Gridded Gaussian model.
"""
import numpy as np
import tensorflow as tf
import pickle

# Load model.
with open('models/ggm_5degree_ct3/ggm.pickle', 'rb') as f:
    m = pickle.load(f)


# Load data and scalers.
data_dir = "models/eddie0205_bs8192_lr5em4/"
datas = []
datas_str = ["X", "XVAL", "Y", "YVAL", "X_", "X_VAL", "Y_", "Y_VAL"]
for i in range(len(datas_str)):
    datas.append(np.load(data_dir + datas_str[i] + '.npy'))
[X, XVAL, Y, YVAL, X_, X_VAL, Y_, Y_VAL] = datas


tf.keras.backend.set_floatx('float64')
ml_model_dir = "models/eddie0205_bs8192_lr5em4/"
# trained_model_file = ml_model_dir + "trained_nn"
trained_model_file = ml_model_dir + "checkpoint_nn_20_4475.83"

scalers = []
scalers_str = ["Xscaler", "Yscaler"]
for i in range(len(scalers_str)):
    with open(ml_model_dir + scalers_str[i] + '.pickle', 'rb') as f:
        scalers.append(pickle.load(f))

[Xscaler, Yscaler] = scalers

# Load neural network and Gaussian mixture layer.
with open(ml_model_dir + 'gm.pickle', 'rb') as f:
    gm = pickle.load(f)
NN = tf.keras.models.load_model(trained_model_file,
                                custom_objects={'nll_reg': gm.nll_reg})
gm.neural_net = NN

# Calculate log Bayes factor for MDN model (eddie2004) against GGM.
# With 45 degree GGM:
# Training data: log K = 384517 = (-20782) - (-405300)
# Testing data:  log K = 343547 = (-75743) - (-419290)
# With 5 degree GGM filling poorly sampled cells with global averages:
# Training data: log K = 15603 = (-20782) - (-36385)
# Testing data:  log K = -7803 = (-75743) - (-67940)
# With 1 degree GGM filling poorly sampled cells with global averages:
# Training data: log K = -189526 = (-20782) - (168743)
# Testing data:  log K = 88362 = (-75743) - (-164106)
# With 10 degree GGM filling poorly sampled cells with global averages:
# Training data: log K = 104566 = (-20782) - (-125348)
# Testing data:  log K = 67855 = (-75743) - (-143598)

# Calculate log Bayes factor for MDN model eddie0205_bs8192_lr5em4 against GGM.
# With 5 degree GGM filling poorly sampled cells with global averages:
# Training data: log K =  = () - ()
# Testing data:  log K =  = () - ()


# Evaluate MDN
ml_nll_ = NN.evaluate(x=X_, y=Y_, batch_size=X_.shape[0])
ml_log_likelihood = -ml_nll_ - X_.shape[0] * np.log(Yscaler.std.prod())

# Evaluate GGM
log_likelihood = m.log_likelihood(XVAL, YVAL)
mean_point_nll = -log_likelihood / X.shape[0]

# Calculate Bayes factor
lnK = ml_log_likelihood - log_likelihood

# Check against globally constant Gaussian model
# log_likelihood_global = m.log_likelihood_based_on_global_averages(Y)
