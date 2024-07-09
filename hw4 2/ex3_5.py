import numpy as np
from scipy.io import loadmat
from scipy.special import expit
import matplotlib.pyplot as plt

np.random.seed(42) 
data = loadmat('data.mat')
X = data['X']
y = data['y'].flatten()  
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_norm = (X - mean) / std
X_with_intercept = np.concatenate([np.ones((X_norm.shape[0], 1)), X_norm], axis=1)
shuffle_indices = np.random.permutation(np.arange(X_with_intercept.shape[0]))
train_indices = shuffle_indices[:int(0.8 * len(shuffle_indices))]
val_indices = shuffle_indices[int(0.8 * len(shuffle_indices)):]
X_train, X_val = X_with_intercept[train_indices], X_with_intercept[val_indices]
y_train, y_val = y[train_indices], y[val_indices]
n = X_train.shape[1]
initial_w = np.zeros(n)
def stochastic_gradient_descent(X, y, initial_w, alpha, lambda_, num_iters):
    m, n = X.shape
    w = initial_w.copy()
    J_history = []
    for iter in range(num_iters):
        cost = 0
        alpha /= num_iters
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index, :].reshape(1, -1)
            yi = y[random_index]
            prediction = expit(xi @ w)
            error = prediction - yi
            gradient = xi.T @ error + (lambda_/m) * np.r_[[0], w[1:]]
            w -= alpha * gradient.flatten()
            cost += - yi * np.log(prediction) - (1 - yi) * np.log(1 - prediction)
        cost /= m
        reg_term = (lambda_ / (2 * m)) * np.sum(w[1:]**2)
        J_history.append(cost + reg_term)
    return J_history
alpha = 1 / 5000
lambda_ = 1 / 5000
num_iters = 1000
J_history_sgd = stochastic_gradient_descent(X_train, y_train, initial_w, alpha, lambda_, num_iters)
plt.plot(J_history_sgd)
plt.xlabel('Num of iterations')
plt.ylabel('Cost J')
plt.title('SGD Convergence with variable step size')
plt.show()

