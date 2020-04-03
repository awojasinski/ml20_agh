import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=210, centers=3, n_features=2, cluster_std=0.5, shuffle=True, random_state=0)
#plt.scatter(X[:,0], X[:,1], c='red', marker='x')

inertia_dif = 100
lastVal = 0
i = 1
kLimit = 20
cost = []
kVal = []
while True:
    km = KMeans(n_clusters=i, init='random', n_init=15, max_iter=400, tol=1e-4, random_state=None)
    km.fit_predict(X)
    cost.append(km.inertia_)
    kVal.append(i)
    if abs(km.inertia_-lastVal) <= inertia_dif or i == kLimit:
        k = i
        break
    i += 1

km = KMeans(n_clusters=k, init='random', n_init=15, max_iter=400, tol=1e-4, random_state=None)
km.fit(X)
print("number of clusters: ", k)

plt.figure(figsize=(6, 5))
plt.plot(kVal, cost, '-*m')
plt.grid()
plt.xlabel('Number of centoids (clusters)')
plt.ylabel('Cost function')
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], s=70, c=km.labels_, cmap=plt.cm.prism)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            marker='*', s=200, color='blue', label='Centers')
plt.legend(loc='best')
plt.xlabel('X0')
plt.ylabel('X1')
plt.show()