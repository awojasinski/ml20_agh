import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import random


def plot_mnist(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.05)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)).T, cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()


DEPTH = 10
dane = loadmat('mnist.mat')

# Podział danych na parametry X oraz odpowiedź y:
X = dane['X']
y = dane['y']

# Standaryzacja
for i in range(X.shape[0]):
    X[i, :] = X[i, :] / np.std(X[i, :])

# Zamiana cyfry 10 -> 0 (błąd w zbiorze danych)
y[np.where(y == 10)] = 0

# Podział na zbiór uczący i testowy (70%, 30%)
features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.3)

# Wysokość i szerokość obrazka z cyfrą
h = 20
w = 20

# Zad 2. Proszę wyświetlić liczbę cyfr oraz liczbę pikseli przypadającą na jeden obraz
print('liczba cyfr w zbiorze danych: ', len(y))
print('ilość pikseli przypadająca na jeden obraz: ', X.shape[1])

random_list = []
titles = []
for i in range(0, 12):
    n = random.randrange(X.shape[0])
    random_list.append(n)
    titles.append(str(y[n]))
# Wyświetlenie przykładowych obrazów z bazy danych
plot_mnist(X[random_list], titles, h=h, w=w)

# Utworzenie instancji klasyfikatora i uczenie na danych treningowych
clf = tree.DecisionTreeClassifier(max_depth=DEPTH)
clf.fit(features_train, labels_train)

y_predicted = clf.predict(features_test)    # Predykcja dla danych testowych

conf_matrix = confusion_matrix(labels_test, y_predicted)
f1 = f1_score(labels_test, y_predicted, average='micro')
clf_report = classification_report(labels_test, y_predicted)

print('----------------------------------------------------------')
for i, confusion in enumerate(conf_matrix):
    print('cyfra %d: %s\n' % (i, confusion))
print('----------------------------------------------------------')
print('Wynik F1 globalny (obliczony z wszystkich TP, FN, FP i TN)\n %0.4f' % f1)
print('----------------------------------------------------------')
print('Raport z klasyfikacji:\n\n', clf_report)
print('----------------------------------------------------------')
