
# Adjusted prior and averaging
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import ot

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

def make_1D_trunc_gauss(n, n_sig, lb, ub, mu, sig):
    x = np.linspace(lb, ub, n)
    normaliser = x[np.argmin(np.abs(x-mu))] # just so to avoid overflow
    h = np.exp(-(x - mu)**2 *n_sig/ (2 * sig**2) + (mu-normaliser)**2 * n_sig/ (2 * sig**2))
    return h/h.sum()
    
def Bary_generator_unif(theta, n_bins=100, n_coef=1000, beta=1, n=4800, reg=1e-2, sig=1):
    bary_list = []
    for i in range(n_coef):
        lb=-0.5 * (i+1)**(-1-2*beta) 
        ub = 0.5*(i+1)**(-1-2*beta)
        i_observations = theta[i]
        hist_array= np.empty((0,n_bins))
        for num in i_observations:
            temp_hist = make_1D_trunc_gauss(n=n_bins, lb=lb , n_sig = n, 
                                            ub=ub , mu=num, sig = sig)
            hist_array = np.vstack((hist_array, temp_hist))
            #hist_array = hist_array[1:,]
            A = hist_array.T
        M = ot.utils.dist0(n_bins)

        M /= M.max()
            
        bary_wass = ot.bregman.barycenter(A, M, reg, weights=None)
        bary_list.append(bary_wass)
    return bary_list

def Bary_list_sampler(Bary_list, beta, n_bins):
    sample = np.zeros(len(Bary_list))
    for i in range(len(Bary_list)):
        lb=-0.5 * (i+1)**(-1-2*beta) 
        ub = 0.5*(i+1)**(-1-2*beta)
        x = np.linspace(lb, ub, n_bins)
        a = np.random.choice(x, p = Bary_list[i])
        sample[i] = a
    return sample

def Bary_generator_gauss(theta, n_bins=100, n_coef=1000, beta=1, n=4800, reg=1e-2, sig=1):
    bary_list = []
    for i in range(n_coef):
        i_observations = theta[i]
        lb = i_observations.min() - 2 * sig/np.sqrt(n + sig**2*i**(1+2*beta))
        ub = i_observations.max() + 2 * sig/np.sqrt(n + sig**2*i**(1+2*beta))
        hist_array= np.empty((0,n_bins))
        for num in i_observations:
            temp_hist = make_1D_trunc_gauss(n=n_bins, lb=lb , n_sig = 1, 
                                            ub=ub , mu=num*n/(n+sig**2*i**(1+2*beta)), 
                                            sig = sig/np.sqrt(n+sig**2*i**(1+2*beta)))
            hist_array = np.vstack((hist_array, temp_hist))
            #hist_array = hist_array[1:,]
        A = hist_array.T
        M = ot.utils.dist0(n_bins)

        M /= M.max()
            
        bary_wass = ot.bregman.barycenter(A, M, reg, weights=None)
        bary_list.append(bary_wass)
    return bary_list