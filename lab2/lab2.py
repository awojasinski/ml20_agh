import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
path = os.getcwd() + '\\dane1.txt'


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


data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
population = data['Population']
profit = data['Profit']

plt.plot(population, profit, 'o')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()

X = np.array([population]).T
X = np.append(np.full(shape=(data.shape[0], 1), fill_value=1), X, axis=1)
y = np.array([profit]).T
theta = np.array([[0, 0]], dtype=float)

print(computeCost(X, y, theta))

iterations = 1000
cost, theta = gradient_prosty(X, y, theta, alpha=0.01, it=iterations)
print(theta)

plt.plot(range(iterations), cost)
plt.title('Cost function')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()



