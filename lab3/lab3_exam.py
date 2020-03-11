import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import math

import os


def sig(t):
    sig = np.empty(shape=(0, 1))
    for i in t:
        sig = np.append(sig,  [[1 / (1 + math.exp(-i))]])
    return sig


def cost(theta, X, y):
    h = np.array([sig(X @ theta.T)]).T
    cost = sum(-y * np.log(h) - (1-y) * np.log(1-h)) / X.shape[0]
    return cost


path = os.getcwd() + '/dane_egz.txt'
data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])

print(data.describe())

X = np.array([data['Exam1'], data['Exam2']],).T
X = np.append(np.full(shape=(X.shape[0], 1), fill_value=1), X, axis=1)
y = np.array([data['Admitted']]).T

'''
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
# TODO
# scatter plot with two groups, 1 - positive 2 - negative
ax.scatter(X[:, 1], X[:, 2], color='green')
ax.set_xlabel('Exam 1')
ax.set_ylabel('Exam 2')
ax.set_title('Rekrutacja')
plt.show()

t = np.arange(-5, 5, 0.5)
plt.plot(t, sig(t))
plt.show()
'''

# Podział na zbiór testowy i treningowy (30%, 70%)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

theta = np.zeros(3)

print(cost(theta, X, y))
