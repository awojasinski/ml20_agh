import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as ch

path = os.getcwd() + '/shopping_data.csv'
customer_data = pd.read_csv(path)
data = customer_data.iloc[:, 3:5].values

plt.figure(figsize=(10, 7))
plt.title("Dendrogram")

dendro= ch.linkage(data, method='ward', metric='euclidean', optimal_ordering=False)
ch.dendrogram(dendro)
plt.show()
