import numpy as np
from scipy.io import loadmat
from scipy.special import expit
import matplotlib.pyplot as plt

np.random.seed(44)  

data = loadmat('data.mat')
X = data['X']
y = data['y'].flatten() 
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_norm = (X - mean) / std

def split_data(X, y, test_size=0.2, random_state=42):
    m = X.shape[0]
    shuffled_indices = np.random.permutation(m)
    test_set_size = int(m * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


alpha = 0.01 / 5000
num_iters = 1000
X_with_intercept = np.concatenate([np.ones((X_norm.shape[0], 1)), X_norm], axis=1)
X_train, X_val, y_train, y_val = split_data(X_with_intercept, y, test_size=0.2)
lambda_ = 1 / 5000

def logistic_regression(X, y, w, alpha, lambda_, num_iters):
    J_history = []
    for _ in range(num_iters):
        z = X.dot(w)
        s = expit(z)
        error = s - y
        gradient = np.dot(X.T, error)
        gradient[1:] += 2 * (lambda_) * w[1:]
        w -= alpha * gradient
        cost = -np.sum(y * np.log(s) + (1 - y) * np.log(1 - s))
        cost += (lambda_) * np.sum(np.square(w[1:]))
        J_history.append(cost)
    return J_history

initial_w = np.zeros(X_with_intercept.shape[1])
J_history = logistic_regression(X_train, y_train, initial_w, alpha, lambda_, num_iters)
plt.plot(J_history)
plt.xlabel('Num of iterations')
plt.ylabel('Cost J')
plt.title('Convergence of cost function over iterations')
plt.show()

