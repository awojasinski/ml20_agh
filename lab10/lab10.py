import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat
from sklearn import metrics


def estimate_gaussian(X):
    """
    Estimates the parameters of a Gaussian distribution using the data in X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.

    Returns
    -------
    mu : ndarray, shape (n_feature,)
        The mean of each feature.
    sigma2 : ndarray, shape (n_feature,)
        The variance of each feature.
    """
    mu = []
    sigma2 = []
    for n in range(X.shape[1]):
        mu.append(np.mean(X[:, n]))
        sigma2.append(np.std(X[:, n])**2)
    return mu, sigma2


def select_threshold(pval, yval):
    f_1 = []
    eps = []
    the_epsilons = np.linspace(np.min(pval), np.max(pval), 10000)
    for epsilon in the_epsilons:
        pred = np.where((pval[:, 0] < epsilon), (np.where((pval[:, 1] < epsilon), 1, 1)), (np.where((pval[:, 1] < epsilon), 1, 0)))
        eps.append(epsilon)
        f_1.append(metrics.f1_score(yval, pred))
    best_f1 = np.max(f_1)
    best_eps_pos = np.argmax(f_1)
    best_epsilon = eps[best_eps_pos]
    return best_epsilon, best_f1


data = loadmat('ex8data1.mat')
X = data['X']

print('Data shape: ', X.shape)

plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('throughput [mb/s]')
plt.ylabel('latency [ms]')
plt.show()

plt.hist(X[:, 0])
plt.title('Histogram of throughput [mb/s]')
plt.show()

plt.hist(X[:, 1])
plt.title('Histogram of latency [ms]')
plt.show()

Xval = data['Xval']
yval = data['yval']

mu, sigma = estimate_gaussian(X)

gauss = np.zeros(X.shape)
pval = np.zeros(Xval.shape)
for i in range(X.shape[1]):
    gauss[:, i] = stats.norm.pdf(X[:, i], mu[i], np.sqrt(sigma[i]))
    pval[:, i] = stats.norm.pdf(Xval[:, i], mu[i], np.sqrt(sigma[i]))

best_eps, best_f1 = select_threshold(pval, yval)
print(best_eps, best_f1)

anomal = np.where(pval < best_eps)
anomal = list(dict.fromkeys(anomal[0]))

plt.figure()
plt.scatter(Xval[:, 0], Xval[:, 1])
plt.scatter(Xval[anomal[0]:anomal[len(anomal)-1], 0], Xval[anomal[0]:anomal[len(anomal)-1], 1], s=10, color='r')
plt.xlabel('Throughput [mb/s]')
plt.ylabel('Latency [ms]')
plt.show()
