import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

path = os.getcwd() + '\\breast_cancer.txt'

dataset = pd.read_csv(path, header=None, names=['ID', 'Clump Thickness', 'Uniformity of Cell Size',
                                                'Uniformity of Cell Shape', 'Marginal Adhesion',
                                                'Single Epithelial Cell Size', 'Bare Nuclei',
                                                'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'])

dataset['Class'].replace(2, 0, inplace=True)
dataset['Class'].replace(4, 1, inplace=True)

for col in dataset.columns:
    mean = dataset[col].describe()['mean']
    dataset[col].fillna(mean, inplace=True)

X = dataset[dataset.columns[1:-1]]
y = dataset[dataset.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

neighbors = np.arange(1, 21)
test_score = []
train_score = []
max_score = 0
opt_n_val = 0
for i, n in enumerate(neighbors):
    kNN = KNeighborsClassifier(n_neighbors=n)
    kNN.fit(X_train, y_train)
    train_score.append(kNN.score(X_train, y_train))
    test_score.append(kNN.score(X_test, y_test))
    if max_score < test_score[i]:
        max_score = test_score[i]
        opt_n_val = n

plt.plot(neighbors, train_score, neighbors, test_score)
plt.grid(True)
plt.title('Skuteczność kNN w zależności od k')
plt.xlabel('k value')
plt.ylabel('Score')
plt.legend(['Train', 'Test'])
plt.show()

kNN = KNeighborsClassifier(n_neighbors=opt_n_val)
kNN.fit(X_train, y_train)
print('Optymalna wartość k: ', opt_n_val)
print('Zbiór treningowy :', kNN.score(X_train, y_train))
print('Zbiór testowy :', kNN.score(X_test, y_test))
