
# Adjusted prior and averaging
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np


def posterior_sample(y_j, n, m, sig, tau, alpha):

    # placeholder for observations
    num_coef = len(y_j)

    def posterior_coef(alpha, n, sig, tau, m, num_coef):
        return 1/(n+sig**2*tau**-1*m*(np.arange(num_coef)+1)**(1+2*alpha))

    theta_j = n*posterior_coef(alpha, n, sig, tau, m, num_coef) * y_j
    var_j = sig**2*m*posterior_coef(alpha, n, sig, tau, m, num_coef)
    dist = tfd.MultivariateNormalDiag(loc=theta_j, scale_diag=np.sqrt(var_j))
    return dist.sample()

def wasserstein_coefs(y_j, n, sig, alpha):
    
    # placeholder for observations
    num_coef = len(y_j)
    
    def posterior_coef(alpha, n, sig, num_coef):
        return 1.0/(n+sig**2.0*(np.arange(num_coef)+1.0)**(1.0+2.0*alpha))
    
    theta_j = n * posterior_coef(alpha, n, sig, num_coef) * y_j
    var_j = sig**2.0* posterior_coef(alpha, n, sig, num_coef)
    return theta_j, var_j

def normal_sampler(mean_vec, var_vec):
    dist = tfd.MultivariateNormalDiag(loc = mean_vec, scale_diag = np.sqrt(var_vec))
    return dist.sample()