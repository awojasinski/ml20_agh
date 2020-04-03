import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

# generacja punktów: 210 próbek, w trzech grupach, po dwie wartości każda próbka
X, y = make_blobs(n_samples=210, centers=3, n_features=2,
                  cluster_std=0.5, shuffle=True, random_state=0)
# wykres wygenerowanych próbek
# plt.scatter(X[:, 0], X[:, 1], c='red', marker='x')

kM = KMeans(n_clusters=3, init='random', n_init=10,
            max_iter=350, tol=1e-4, random_state=None)
kM.fit(X)

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], s=70, c=kM.labels_, cmap=plt.cm.prism)
plt.scatter(kM.cluster_centers_[:, 0], kM.cluster_centers_[:, 1],
            marker='*', s=200, color='blue', label='Centers')
plt.legend(loc='best')
plt.xlabel('X0')
plt.ylabel('X1')
plt.show()

print('Inertia: ', kM.inertia_)

X, y = make_blobs(n_samples=210, centers=5, n_features=2,
                  cluster_std=0.8, shuffle=True, random_state=0)
# wykres wygenerowanych próbek
# plt.scatter(X[:, 0], X[:, 1], c='red', marker='x')

kM = KMeans(n_clusters=2, init='random', n_init=10,
            max_iter=350, tol=1e-4, random_state=None)
kM.fit(X)

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], s=70, c=kM.labels_, cmap=plt.cm.prism)
plt.scatter(kM.cluster_centers_[:, 0], kM.cluster_centers_[:, 1],
            marker='*', s=200, color='blue', label='Centers')
plt.legend(loc='best')
plt.xlabel('X0')
plt.ylabel('X1')
plt.show()

print('Inertia: ', kM.inertia_)
