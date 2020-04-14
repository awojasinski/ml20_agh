import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering

path = os.getcwd() + '/shopping_data.csv'
customer_data = pd.read_csv(path)

data = customer_data.iloc[:, 3:5].values

n_clusters = 5
linkage_list = ['single', 'average', 'complete', 'ward']
for l in linkage_list:
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage=l)
    cluster_labels = clusterer.fit(data)

    plt.figure()
    colors = cm.nipy_spectral(cluster_labels.labels_.astype(float) / n_clusters)
    plt.scatter(data[:, 0], data[:, 1], marker='.', s=70, lw=0, alpha=0.7, c=colors, edgecolors='k')
    plt.title(l)
    plt.xlabel("value 1")
    plt.ylabel("value 2")
    plt.show()
