import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
path = os.getcwd() + '\\dane1.txt'


def computeCost(X, y, theta):
    h = X @ theta.T
    cost = sum(np.square(h - y)) / (X.shape[0] * 2)

    return cost

# TODO:
# 1. add animation (every x iterations show actual line)
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
plt.grid(True)
plt.show()

X = np.array([population]).T
X = np.append(np.full(shape=(data.shape[0], 1), fill_value=1), X, axis=1)
y = np.array([profit]).T
theta = np.array([[0, 0]], dtype=float)

print(computeCost(X, y, theta))

iterations = 1000
cost, theta = gradient_prosty(X, y, theta, alpha=0.01, it=iterations)
print(theta)

plt.plot(range(len(cost)), cost)
plt.title('Cost function')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.grid(True)
plt.show()

sorted_X = X[np.argsort(X[:, 1])]
plt.plot(population, profit, 'o', label='data')
plt.plot(sorted_X[:, 1], sorted_X @ theta.T, label='linear regresion')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.grid(True)
plt.legend()
plt.show()

# TODO:
# 3D contour of cost function for theta from plane
'''
theta = np.array([np.linspace(-10, 10, 1000).T, np.linspace(-10, 10, 1000).T]).T
XX, YY = np.meshgrid(theta[:, 0], theta[:, 1])
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(XX, YY, Z, 50, cmap='binary')
ax.set_xlabel('theta 0')
ax.set_ylabel('theta 1')
ax.set_zlabel('cost')
'''
