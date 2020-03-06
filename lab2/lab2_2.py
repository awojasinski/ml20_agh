import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
path = os.getcwd() + '\\dane2.txt'


def computeCost(X, y, theta):
    h = X @ theta.T
    cost = sum(np.square(h - y)) / (X.shape[0] * 2)

    return cost


def gradient_prosty(X, y, theta, alpha, it):
    cost = np.empty(shape=([0, 1]))
    dt_theta = np.empty(shape=(theta.size,), dtype=float)

    for i in range(it):
        h = X @ theta.T
        for j in range(theta.size):
            dt_theta[j] = (h - y).T @ X[:, j] / X.shape[0]
        theta -= alpha * dt_theta
        cost = np.append(cost, [computeCost(X, y, theta)])

    return cost, theta


def normalize_data(X):
    X_mean = np.mean(X, axis=0)
    X_mean_M = np.full(X.shape, fill_value=X_mean)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean_M) / X_std
    return X_norm


data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

house_size = data['Size']
bedrooms = data['Bedrooms']
house_price = data['Price']
np_data = data.to_numpy()

np_data_norm = normalize_data(np_data)

X = np.array(np_data_norm[:, 0:2])
X = np.append(np.full(shape=(X.shape[0], 1), fill_value=1), X, axis=1)
y = np.array(np_data_norm[:, 2]).T
theta = np.array([0, 0, 0], dtype=float)

print(computeCost(X, y, theta))

iterations = 1000
cost, theta = gradient_prosty(X, y, theta, alpha=0.01, it=iterations)

plt.plot(range(len(cost)), cost)
plt.title('Cost function')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.grid(True)
plt.show()

