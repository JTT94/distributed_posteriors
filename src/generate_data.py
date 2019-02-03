
# import stuff
import numpy as np


def sample_prior(num_coef=200, tau=1, beta=1.5):
    theta = tau * (np.random.random(num_coef) * 2 - 1) / (np.arange(1, num_coef + 1, 1) ** (beta + 0.5))
    return theta


def generate_data_from_signal(theta, sig, m, n):
    noise_sig = np.sqrt(sig**2*m/n)
    return theta + noise_sig*np.random.random(len(theta))


def generate_grouped_data(signal, n=200, sig=1, m=40):
    return np.array([generate_data_from_signal(signal, sig, m, n)for _ in range(m)])


def generate_series(theta):
    N = len(theta)
    X = np.arange(0, 1, 0.01)
    Y = [np.sum(theta[:N // 2] * np.cos(2 * np.pi * (np.arange(1, N // 2 + 1, 1) * x)))
         + np.sum(theta[N // 2:] * np.sin(2 * np.pi * (np.arange(1, N // 2 + 1, 1) * x))) for x in X]
    return X,Y


