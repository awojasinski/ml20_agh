import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs, make_moons, make_circles

x, y = make_moons(n_samples=200, noise=.05, random_state=0)
n_clusters = 2

# klasteryzacja
clusterer = KMeans(n_clusters=n_clusters, init='random', random_state=10)
cluster_labels = clusterer.fit_predict(x, y)

plt.figure()
colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
plt.scatter(x[:, 0], x[:, 1], marker='.', s=70, lw=0, alpha=0.7, c=colors, edgecolor='k')
plt.title('KMeans')
plt.show()

# klasteryzacja hierarchiczna
linkage_list = ['single', 'average', 'complete', 'ward']
for linkage in linkage_list:
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage=linkage)
    clustering_labels = clustering_model.fit_predict(x)

    plt.figure()
    colors = cm.nipy_spectral(clustering_labels.astype(float) / n_clusters)
    plt.scatter(x[:, 0], x[:, 1], marker='.', s=70, lw=0, alpha=0.7, c=colors, edgecolor='k')

    plt.title(linkage)
    plt.xlabel("Value 1")
    plt.ylabel("Value 2")
    plt.show()

# DBSCAN
plt.figure()
plt.title('DBSCAN')
dbscan = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
dbscan_labels = dbscan.fit_predict(x)
colors = cm.nipy_spectral(dbscan_labels.astype(float) / n_clusters)
plt.scatter(x[:, 0], x[:, 1], marker='.', s=70, lw=0, alpha=0.7, c=colors, edgecolor='k')
plt.show()
