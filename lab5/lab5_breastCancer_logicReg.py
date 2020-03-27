import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

logreg = LogisticRegression(solver='saga', penalty='l1', C=1)
logreg.fit(X_train, y_train)

print('Zbiór treningowy :', logreg.score(X_train, y_train))
print('Zbiór testowy :', logreg.score(X_test, y_test))

# Wykres skuteczności algorytmu względem L2 od wartości C
C_val = np.linspace(0.0001, 1, 10)
test_score = []
train_score = []

for c in C_val:
    logreg = LogisticRegression(solver='saga', penalty='l2', C=c)
    logreg.fit(X_train, y_train)
    train_score.append(logreg.score(X_train, y_train))
    test_score.append(logreg.score(X_test, y_test))

plt.plot(C_val, train_score, C_val, test_score)
plt.grid(True)
plt.title('Skuteczność regresji logicznej w zależności od wartości C')
plt.xlabel('C value')
plt.ylabel('Score')
plt.legend(['Train', 'Test'])
plt.show()
