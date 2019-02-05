
# import stuff
import numpy as np


def sample_prior(num_coef=200, tau=1, beta=1.5):
    theta = np.sqrt(tau) * (np.random.normal(size=num_coef)) / (np.arange(1, num_coef + 1, 1) ** (beta + 0.5))
    return theta


def generate_data_from_signal(theta, sig, m, n):
    noise_sig = np.sqrt(sig**2*m/n)
    return theta + noise_sig*np.random.normal(size=len(theta))


def generate_grouped_data(signal, n=200, sig=1, m=40):
    return np.array([generate_data_from_signal(signal, sig, m, n)for _ in range(m)])


def generate_series(theta):
    N = len(theta)
    X = np.arange(0, 1, 1/len(theta))
    Y = [np.sum(theta[:N // 2] * np.cos(2 * np.pi * (np.arange(1, N // 2 + 1, 1) * x)))
         + np.sum(theta[N // 2:] * np.sin(2 * np.pi * (np.arange(1, N // 2 + 1, 1) * x))) for x in X]
    return X,Y


def generate_theta(num_coeff = 1000):
    N = num_coeff
    theta = (0.21+np.cos(np.arange(1.0,N+1.0,1.0)) * 0.205)  / (
        np.arange(1.0,N+1.0,1.0))**1.5 
    return theta


def generate_alternate_series(theta):
    N = len(theta)
    X = np.arange(0,1,1/len(theta))
    Y = [np.sum(theta * np.cos(np.pi * (np.arange(1.0,N+1.0,1.0) - 0.5) * x)) for x in X]
    return X,Y 


