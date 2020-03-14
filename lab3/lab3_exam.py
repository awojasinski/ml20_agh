import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import math

import os


def normalize_data(X):
    X_mean = np.mean(X, axis=0)
    X_mean_M = np.full(X.shape, fill_value=X_mean)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean_M) / X_std
    return X_norm


def sig(t):
    sig = np.empty(shape=(0, 1), dtype=float)
    for i in t:
        sig = np.append(sig,  [[1 / (1 + math.exp(-i))]])
    return sig


def compute_cost(theta, X, y):
    h = np.array([sig(X @ theta.T)]).T
    cost = sum(-y * np.log(h) - (1-y) * np.log(1-h)) / X.shape[0]
    return cost


def gradient_decent(X, y, theta, alpha, it):
    cost = np.empty(shape=([0, 1]))
    dt_theta = np.empty(shape=(theta.size,), dtype=float)
    for i in range(it):
        h = sig(X @ theta.T)
        for j in range(theta.size):
            dt_theta[j] = sum((h - y.T) @ X[:, j]) / X.shape[0]
        theta -= alpha * dt_theta
        cost = np.append(cost, [compute_cost(theta, X, y)])
    return theta, cost


path = os.getcwd() + '/dane_egz.txt'
data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])
print('-----------------------------------------------------------------------------')
print('Opis analizowanych danych')
print('-----------------------------------------------------------------------------')
print(data.describe())
print('-----------------------------------------------------------------------------')

X = np.array([data['Exam1'], data['Exam2']],).T
X = normalize_data(X)
X = np.append(np.full(shape=(X.shape[0], 1), fill_value=1), X, axis=1)
y = np.array([data['Admitted']]).T

X_positive = np.argwhere(y == 1)[:, 0]
X_negative = np.argwhere(y == 0)[:, 0]

plt.scatter(X[X_positive, 1], X[X_positive, 2], color='green', label='Przyjęty')
plt.scatter(X[X_negative, 1], X[X_negative, 2], color='red', label='Nieprzyjęty')
plt.xlabel('Exam 1')
plt.ylabel('Exam 2')
plt.title('Rekrutacja')
plt.legend()
plt.show()

t = np.arange(-5, 5, 0.5)
plt.plot(t, sig(t))
plt.grid(True)
plt.show()

# Podział na zbiór testowy i treningowy (30%, 70%)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

theta = np.zeros(3)
print('Funkcja kosztu dla theta = [0, 0, 0]: ', compute_cost(theta, X_train, Y_train))

alpha = 1
it = 150
theta, cost = gradient_decent(X_train, Y_train, theta, alpha=alpha, it=it)
print('-----------------------------------------------------------------------------')
print('Metoda gradientu prostego wyznaczenia współczynników theta regresji logicznej')
print('Współczynniki:\nalpha = %f\nliczba iteracji = %d' %(alpha, it))
print('-----------------------------------------------------------------------------')
print('Funkcja kosztu: ', cost[-1])
print('Wartości theta: ', theta)
print('-----------------------------------------------------------------------------')
plt.plot(range(it), cost)
plt.show()

Y_result = sig(X_test @ theta.T)
Y_result = np.where(Y_result > 0.5, Y_result, 0)
Y_result = np.where(Y_result <= 0.5, Y_result, 1)

indexes_error = np.argwhere(Y_result != Y_test.T)

X_positive = np.argwhere(Y_test == 1)[:, 0]
X_negative = np.argwhere(Y_test == 0)[:, 0]

accuracy = (len(Y_test) - len(indexes_error)) / len(Y_test)
print('Skuteczność algorytmu: %.2f%s' %(accuracy, '%'))

plt.scatter(X_test[X_positive, 1], X_test[X_positive, 2], color='green', label='Przyjęty')
plt.scatter(X_test[X_negative, 1], X_test[X_negative, 2], color='red', label='Nieprzyjęty')
plt.scatter(X_test[indexes_error, 1], X_test[indexes_error, 2], color='black', label='Źle rozpoznane')
plt.xlabel('Exam 1')
plt.ylabel('Exam 2')
plt.title('Rekrutacja')
plt.legend()
plt.show()
