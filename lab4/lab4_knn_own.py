from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
import numpy as np


def kNN(features_train, features_test, labels_train, k):
    # Inicjalizacja tablicy klasyfikacji
    predictions = np.empty(shape=(0, 1))

    # Obliczenie ilości występujących klas
    labels = len(np.unique(labels_train))

    for feature in features_test:
        neighbors = np.zeros(shape=(labels, 1))     # Inicjalizacja tablicy liczącej sąsiadów
        dist = np.zeros(shape=(features_train.shape[0],))   # Inicjalizacja tablicy odległości
        x = np.arange(features_train.shape[0])
        dist = np.append([x], [dist], axis=0).T
        for n, i in enumerate(features_train):
            dist[n, 1] = distance.euclidean(feature, i)     # Obliczenie odległości
        dist = dist[dist[:, 1].argsort()]   # Posortowanie rosnąco odległości
        for j in range(k):
            neighbors[int(labels_train[int(dist[j, 0])])] += 1
        predictions = np.append(predictions, np.argmax(neighbors))

    return predictions


iris = datasets.load_iris()

# Podział na zbiór uczący i testowy (70%, 30%)
features_train, features_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size=0.3)

predictions = kNN(features_train, features_test, labels_train, 3)

# Sprawdzanie skuteczności klasyfikatora
output = accuracy_score(labels_test, predictions)
print('Accuracy: ', output)
