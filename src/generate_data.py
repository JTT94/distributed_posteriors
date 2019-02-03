
# import stuff
import numpy as np


def sample_prior(num_coef=200, tau=1, beta=1.5):
    theta = tau * (np.random.random(num_coef) * 2 - 1) / (np.arange(1, num_coef + 1, 1) ** (beta + 0.5))
    return theta


def generate_data_from_signal(theta, sig, m, n):
    noise_sig = np.sqrt(sig**2*m/n)
    return theta +  noise_sig*np.random.random(len(theta))


def generate_grouped_data(num_coef=100, n=200, sig=1, m=40, tau=1, beta=1.5):
    signal = sample_prior(num_coef, tau, beta)
    return np.array([generate_data_from_signal(signal, sig, m, n)for _ in range(m)])


def generate_series(theta):
    N = len(theta)
    X = np.arange(0, 1, 0.01)
    Y = [np.sum(theta[:N // 2] * np.cos(2 * np.pi * (np.arange(1, N // 2 + 1, 1) * x)))
         + np.sum(theta[N // 2:] * np.sin(2 * np.pi * (np.arange(1, N // 2 + 1, 1) * x))) for x in X]
    return X,Y


# from matplotlib import pyplot as plt
# data = generate_grouped_data(num_coef=2)
# X, Y = generate_series(data[0])
# fig, ax = plt.subplots(1,2, figsize=(20/1,9/1))
# a = ax[0]
# a.plot(X,Y)