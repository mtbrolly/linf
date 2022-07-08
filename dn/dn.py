"""
Module for creating density network models.

Classes:

    Gm
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import skew, kurtosis
tfd = tfp.distributions
tf.keras.backend.set_floatx('float64')


class Gm():
    """
    Class for creating a Gaussian mixture density network from a nn.
    """

    def __init__(self,
                 o_size=None,
                 n_c=1,
                 scale_type='tril',
                 neural_net=None,
                 batch_size=None,
                 reg_c=(None, None, None)):
        """
        Parameters
        ----------

        o_size : int
            Size of y.
        n_c : int
            Number of mixture components.
        scale_type : str
            Either 'diag' or 'tril', so that mixture components are
            MultivariateNormalDiag or MultivariateNormalTriL distributions,
            resp..
        nn : keras.engine.functional.Functional
            Neural network whose output provides the mixture parameters for y|x
            given input x.
        reg_c : length-3 tuple, optional
            Regularisation coefficients for mixture parameters,
            (probs, loc, scale).

        """

        self.o_size = o_size
        self.n_c = n_c
        self.scale_type = scale_type
        if self.scale_type == 'tril':
            def calc_n_cov_elements(o_size):
                return o_size * (o_size + 1) // 2

            self.n_cov_elements = calc_n_cov_elements(self.o_size) * self.n_c
            self.params_size = ((self.o_size + 1) * self.n_c
                                + self.n_cov_elements)
        elif self.scale_type == 'diag':
            self.params_size = (2 * self.o_size + 1) * self.n_c
        self.neural_net = neural_net
        self.batch_size = batch_size

        self.reg_c = reg_c

    def get_params_from_nn_out(self, nn_out):
        """
        Takes raw nn output and returns mixture parameters.
        """
        if self.scale_type == 'diag':
            nn_out = tf.reshape(
                nn_out, (nn_out.shape[0], self.n_c, -1))
            nn_out_splits = tf.split(nn_out, [1, self.o_size,
                                              self.o_size], axis=-1)
            probs = tf.keras.activations.softmax(nn_out_splits[0][..., 0],
                                                 axis=-1)
            loc = nn_out_splits[1]
            scale_diag = tf.keras.activations.softplus(nn_out_splits[2])
            return probs, loc, scale_diag
        # elif self.scale_type == 'tril':
        nn_out = tf.reshape(
            nn_out, (nn_out.shape[0], self.n_c, -1))
        nn_out_splits = tf.split(nn_out, [1, self.o_size,
                                          self.n_cov_elements // self.n_c],
                                 axis=-1)
        probs = tf.keras.activations.softmax(nn_out_splits[0][..., 0],
                                             axis=-1)
        loc = nn_out_splits[1]
        scale_tril_flat = nn_out_splits[2]
        scale_tril = tfp.math.fill_triangular(scale_tril_flat)
        scale_tril = tensor_diag_softplus(scale_tril)
        return probs, loc, scale_tril

    def get_gms_from_params(self, probs, loc, scale_param):
        """
        Takes mixture parameters and returns the corresponding Tensorflow-
        Probability mixture distributions.
        """
        if self.scale_type == 'diag':
            gms = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(
                    probs=probs),
                components_distribution=tfd.MultivariateNormalDiag(
                    loc=loc,
                    scale_diag=scale_param))
            return gms

        gms = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=probs),
            components_distribution=tfd.MultivariateNormalTriL(
                loc=loc,
                scale_tril=scale_param))
        return gms

    def get_gms_from_nn_out(self, nn_out):
        """
        Takes raw nn output and returns Tensorflow-Probability mixture
        distributions.
        """
        params = self.get_params_from_nn_out(nn_out)
        gms = self.get_gms_from_params(*params)
        return gms

    def get_params_from_x(self, x):
        """
        Takes x and returns mixture parameters.
        """
        if not self.batch_size:
            self.batch_size = self.neural_net.layers[0].input_shape[0][0]
        nn_out = self.neural_net.predict(x)  # , batch_size=self.batch_size)
        probs, loc, scale_param = self.get_params_from_nn_out(nn_out)
        return probs, loc, scale_param

    def get_gms_from_x(self, x):
        """
        Takes x and returns """
        params = self.get_params_from_x(x)
        gms = self.get_gms_from_params(*params)
        return gms

    def nll(self, y_true, nn_out, return_params=False):
        """
        Loss function: negative-log-likelihood.
        """
        params = self.get_params_from_nn_out(nn_out)
        gms = self.get_gms_from_params(*params)
        log_probs = gms.log_prob(y_true)
        log_prob = tf.math.reduce_sum(log_probs, axis=-1)
        if return_params:
            return -log_prob, params
        return -log_prob

    def nll_reg(self, y_true, nn_out):
        """
        Loss function: negative-log-likelihood plus regularisation of mixture
        parameters.
        """
        nll, params = self.nll(y_true, nn_out, return_params=True)
        loss = nll
        if self.reg_c[0]:
            probs = params[0]
            probs_reg = -self.reg_c[0] * tf.math.reduce_sum(
                tf.math.multiply(probs, tf.math.log(probs + 1e-14)))
            loss += probs_reg
        if self.reg_c[2]:
            scale = params[2]
            scale_reg = self.reg_c[2] * tf.math.log(
                tf.math.reduce_sum(scale))
            loss += scale_reg
        return loss

    def sample(self, x, sample_shape=(1,)):
        """
        Takes x and returns samples of y|x with shape sample_shape.
        """
        gms = self.get_gms_from_x(x)
        return gms.sample(sample_shape=sample_shape)

    def mixture_entropy(self, x):
        """
        Takes x and returns the entropy (in nats) of the discrete distribution
        part of the mixture.
        N.B. maximum entropy of log(n) is attained by uniform distribution.
        """
        gms = self.get_gms_from_x(x)
        alphas = gms.mixture_distribution.probs
        entropy = tf.math.reduce_sum(
            tf.math.multiply(-alphas, tf.math.log(alphas)), axis=-1)
        return entropy

    def entropy(self, x, sample_size, block_size=None):
        if not block_size:
            gms = self.get_gms_from_x(x)
            ys = gms.sample((sample_size, ))
            log_prob_ys = gms.log_prob(ys)
            entropies = tf.math.reduce_mean(-log_prob_ys, axis=0)
        else:
            x_size = x.shape[0]
            entropies = np.zeros((x_size,))
            for i in range(int(np.ceil(x_size / block_size))):
                gms = self.get_gms_from_x(
                    x[i * block_size: (i + 1) * block_size, :])
                ys = gms.sample((sample_size, ))
                log_prob_ys = gms.log_prob(ys)
                entropies[i * block_size: (i + 1) * block_size] = (
                    tf.math.reduce_mean(-log_prob_ys, axis=0))
        return entropies

    def skewness(self, x, sample_size, block_size=None):
        if not block_size:
            gms = self.get_gms_from_x(x)
            ys = gms.sample((sample_size, ))
            skewness = skew(ys, axis=0)
        else:
            x_size = x.shape[0]
            skewness = np.zeros((x_size, 2))
            for i in range(int(np.ceil(x_size / block_size))):
                gms = self.get_gms_from_x(
                    x[i * block_size: (i + 1) * block_size, :])
                ys = gms.sample((sample_size, ))
                skewness[i * block_size: (i + 1) * block_size] = (
                    skew(ys, axis=0))
        return skewness

    def kurtosis(self, x, sample_size, block_size=None):
        if not block_size:
            gms = self.get_gms_from_x(x)
            ys = gms.sample((sample_size, ))
            kurt = kurtosis(ys, axis=0, fisher=False)
        else:
            x_size = x.shape[0]
            kurt = np.zeros((x_size, 2))
            for i in range(int(np.ceil(x_size / block_size))):
                gms = self.get_gms_from_x(
                    x[i * block_size: (i + 1) * block_size, :])
                ys = gms.sample((sample_size, ))
                kurt[i * block_size: (i + 1) * block_size] = (
                    kurtosis(ys, axis=0, fisher=False))
        return kurt

    def S3(self, x, sample_size, block_size=None):
        if not block_size:
            gms = self.get_gms_from_x(x)
            ys = gms.sample((sample_size, ))
            S3l = tf.math.reduce_mean(ys[..., 0] ** 3, axis=0)
            S3t = tf.math.reduce_mean(ys[..., 0] * ys[..., 1] ** 2, axis=0)
        else:
            x_size = x.shape[0]
            S3l = np.zeros((x_size, 2))
            S3t = np.zeros((x_size, 1))
            for i in range(int(np.ceil(x_size / block_size))):
                gms = self.get_gms_from_x(
                    x[i * block_size: (i + 1) * block_size, :])
                ys = gms.sample((sample_size, ))
                S3l[i * block_size: (i + 1) * block_size] = (
                    tf.math.reduce_mean(ys ** 3, axis=0))
                S3t[i * block_size: (i + 1) * block_size] = (
                    tf.math.reduce_mean(
                        ys[..., 0:1] * ys[..., 1:2] ** 2, axis=0))
        return S3l, S3t

    def density(self, x, y):
        """
        Takes x and y and returns probability density of y|x.
        """
        gms = self.get_gms_from_x(x)
        return gms.prob(y)

    def log_density(self, x, y):
        """
        Takes x and y and returns log probability density of y|x.
        """
        gms = self.get_gms_from_x(x)
        return gms.log_prob(y)

    def get_marginal_gms_from_gms(self, gms, y_ind):
        """
        Takes a multivariate Gaussian mixture which models y|x and y_ind (index
        of a component of y) and returns the marginal (scalar) Gaussian mixture
        which models y[y_ind]|x.
        """
        probs = gms.mixture_distribution.probs
        marg_loc = gms.components_distribution.loc[..., y_ind]
        marg_scale = gms.components_distribution.covariance()[...,
                                                              y_ind, y_ind]
        marginal_gms = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=probs),
            components_distribution=tfd.Normal(
                loc=marg_loc,
                scale=marg_scale))
        return marginal_gms

    def get_marginal_gms_from_x(self, x, y_ind):
        """
        Takes x and y_ind (index of a component of y) and returns the marginal
        (scalar) Gaussian mixture modelling y[y_ind]|x, derived from the
        multivariate Gaussian mixture which models y|x.
        """
        gms = self.get_gms_from_x(x)
        marginal_gms = self.get_marginal_gms_from_gms(gms, y_ind)
        return marginal_gms

    def log_marg_density(self, x, y, y_ind):
        """
        Takes x, y and y_ind
        """
        marginal_gms = self.get_marginal_gms_from_x(x, y_ind)
        return marginal_gms.log_prob(y)


def tensor_diag_softplus(tensor):
    """
    Utility function for applying softmax activation to the diagonal of a batch
    of covariance matrices.
    """
    diag = tf.linalg.diag_part(tensor)
    transformed_diag = tf.keras.activations.softplus(diag)
    transformed_tensor = tf.linalg.set_diag(tensor, transformed_diag)
    return transformed_tensor
