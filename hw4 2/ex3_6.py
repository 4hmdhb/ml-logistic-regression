import numpy as np
from scipy.io import loadmat
from scipy.special import expit
from sklearn.svm import SVC
import csv
np.random.seed(42)  
data = loadmat('data.mat')
X = data['X']
y = data['y'].flatten() 
X_test = data['X_test']
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_norm = (X - mean) / std
X_test_norm = (X_test - mean) / std
X_with_intercept = np.concatenate([np.ones((X_norm.shape[0], 1)), X_norm], axis=1)
X_test_with_intercept = np.concatenate([np.ones((X_test_norm.shape[0], 1)), X_test_norm], axis=1)
shuffle_indices = np.random.permutation(np.arange(X_with_intercept.shape[0]))
train_indices = shuffle_indices[:int(0.8 * len(shuffle_indices))]
val_indices = shuffle_indices[int(0.8 * len(shuffle_indices)):]
X_train, X_val = X_with_intercept[train_indices], X_with_intercept[val_indices]
y_train, y_val = y[train_indices], y[val_indices]
n = X_train.shape[1]
initial_theta = np.zeros(n)
def stochastic_gradient_descent(X, y, initial_w, alpha, lambda_, num_iters):
    m, n = X.shape
    w = initial_w.copy()
    J_history = []
    iter_number = 1
    for iter in range(num_iters):
        cost = 0
        for i in range(m):
            alpha = alpha/(iter + 1)
            random_index = np.random.randint(m)
            xi = X[random_index, :].reshape(1, -1)
            yi = y[random_index]
            prediction = expit(xi @ w)
            error = prediction - yi
            gradient = xi.T @ error + (lambda_/m) * np.r_[[0], w[1:]]
            w -= alpha * gradient.flatten()
            cost += - yi * np.log(prediction) - (1 - yi) * np.log(1 - prediction)
            iter_number += 1        
        cost /= m
        reg_term = (lambda_ / (2 * m)) * np.sum(w[1:]**2)
        J_history.append(cost + reg_term)
    return w, J_history
alpha = 0.1  
lambda_ = 1 
num_iters = 100 
theta_sgd, J_history_sgd = stochastic_gradient_descent(X_train, y_train, initial_theta, alpha, lambda_, num_iters)
def predict(theta, X):
    probabilities = expit(X.dot(theta))
    return [1 if x >= 0.5 else 0 for x in probabilities]
predictions = predict(theta_sgd, X_test_with_intercept)
with open('example.csv', mode='w', newline='\n') as file:
    writer = csv.writer(file)
    writer.writerow(["Id", "Category"])
    id = 1
    for categ in predictions:
        writer.writerow([id, categ])
        id += 1

