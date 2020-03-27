import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
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

tree_depth = np.arange(1, 26)
test_score = []
train_score = []
max_score = 0
opt_depth = 0
for i, d in enumerate(tree_depth):
    clf = tree.DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train, y_train)
    train_score.append(clf.score(X_train, y_train))
    test_score.append(clf.score(X_test, y_test))
    if max_score < test_score[i]:
        max_score = test_score[i]
        opt_depth = d

plt.plot(tree_depth, train_score, tree_depth, test_score)
plt.grid(True)
plt.title('Skuteczność drzewa w zależności od głebokości')
plt.xlabel('depth value')
plt.ylabel('Score')
plt.legend(['Train', 'Test'])
plt.show()

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
print('Optymalna głębokość: ', opt_depth)
print('Zbiór treningowy :', clf.score(X_train, y_train))
print('Zbiór testowy :', clf.score(X_test, y_test))

