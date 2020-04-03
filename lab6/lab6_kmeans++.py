import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import sklearn.cluster as clust

X, y = make_blobs(n_samples=210, centers=3, n_features=2, cluster_std=0.5, shuffle=True, random_state=0)
#plt.scatter(X[:,0], X[:,1], c='red', marker='x')

cost_list = []
k_val = range(2, 16)
for i in k_val:
    kmean = clust.KMeans(n_clusters=i, init='k-means++', n_init=15, max_iter=400, tol=1e-4, random_state=None)
    kmean.fit_predict(X)
    cost_list.append(kmean.inertia_)

plt.figure(figsize=(6, 5))
plt.plot(k_val, cost_list, '-*m')
plt.grid()
plt.xlabel('Number of centoids (clusters)')
plt.ylabel('Cost function')
plt.show()