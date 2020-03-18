from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

# Podział na zbiór uczący i testowy (70%, 30%)
features_train, features_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size=0.3)

# Utworzenie i wyuczenie klasyfikatora
kNN = KNeighborsClassifier(n_neighbors=3)
kNN.fit(features_train, labels_train)

# Predykcja wartości
predictions = kNN.predict(features_test)

# Sprawdzanie skuteczności klasyfikatora
output = accuracy_score(labels_test, predictions)
print('Accuracy: ', output)
