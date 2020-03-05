import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

path = os.getcwd() + '\\dane1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])


def computeCost(X, y, theta):
    h = X @ np.transpose(theta)
    diff_yk = h - y
    square = np.power(diff_yk, 2)
    cost =  sum(square)/(len(y)*2)
    
    return cost


def gradient_prosty(X, y, theta, alpha, it):
    #cost = np.empty()

    for i in range(it):
        h = X @ np.transpose(theta)
        dt_theta_0 = sum(h-y) / len(y)
        dt_theta_1 = sum((h - y) * X) / len(y)
        print(dt_theta_0.shape)
        print(dt_theta_1.shape)
        theta[0] -= alpha @ dt_theta_0
        theta[1] -= alpha @ dt_theta_1
        cost.append(computeCost(X, y, theta))

        return cost, theta


'''
print(data.head)
print(data.describe())
'''

#data = data.sort_values('Population')
'''
plt.plot(data['Population'], data['Profit'], 'o')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.grid(True)
plt.show()
'''

data.insert(0, "No.", range(data.shape[0]), True)

X = data[['No.', 'Population']].copy()
y = data['Profit'].copy()

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

print(computeCost(X, np.transpose(y), theta))
cost, theta = gradient_prosty(X, np.transpose(y), theta, alpha=0.01, it=1000) 
plt.plot_surface(theta[0], theta[1], cost)
plt.show()