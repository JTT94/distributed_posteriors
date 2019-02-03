
# Adjusted prior and averaging
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
x = tf.Variable(np.arange(100), name='var', dtype='float32')

first_batch = tf.slice(x, [0], [50])
mean1 = tf.reduce_mean(first_batch)
dist = tfd.MultivariateNormalFullCovariance(loc=[mean1, mean1], covariance_matrix=[[11, 3.], [4., 22.]])


# posterior for each coordinate, parameters
n = 100
m = 40
sig =1
tau = 1
alpha = 1.5

# placeholder for observations
y_j = np.random.random(100)
num_coef = len(y_j)

def posterior_coef(alpha, n, sig, tau, m, num_coef):
    return 1/(n+sig**2*tau**-1*m*(np.arange(num_coef)+1)**(1+2*alpha))


theta_j = n*posterior_coef(alpha, n, sig, tau, m, num_coef) * y_j
var_j = sig**2*m*posterior_coef(alpha, n, sig, tau, m, num_coef)
dist = tfd.MultivariateNormalDiag(loc=theta_j, scale_diag=np.sqrt(var_j))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(dist.sample()))

with tf.device("/job:local/task:0"):
    second_batch = tf.slice(x, [50], [-1])