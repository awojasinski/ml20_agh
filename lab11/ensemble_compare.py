import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 1

np.set_printoptions(precision=2)

#%% Load the dataset.
wine = datasets.load_wine()
X, y = wine.data, wine.target

skf = StratifiedKFold(n_splits=5)

# Define classifiers and classifier ensembles.
# Compare performance of the models.
clf = DecisionTreeClassifier(min_samples_leaf=3, max_depth=1, random_state=RANDOM_STATE)
val = []
for train_data, test_data in skf.split(X, y):
    X_train, X_test = X[train_data], X[test_data]
    y_train, y_test = y[train_data], y[test_data]

    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    test = np.where(predict == y_test, 1, 0)
    test = np.sum(test)/len(y_test)
    val.append(test)

avg = np.sum(val)/len(val)
print('Decision tree scores: ', val, '(avg = ', avg, ')\n')


clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(min_samples_leaf=3, max_depth=1, random_state=RANDOM_STATE),n_estimators=50)
val = []
for train_data, test_data in skf.split(X, y):
    X_train, X_test = X[train_data], X[test_data]
    y_train, y_test = y[train_data], y[test_data]

    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    test = np.where(predict == y_test, 1, 0)
    test = np.sum(test) / len(y_test)
    val.append(test)

avg = np.sum(val) / len(val)
print('Bagging scores: ', val, '(avg = ', avg, ')\n')


clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_leaf=3, max_depth=1, random_state=RANDOM_STATE), algorithm='SAMME', n_estimators=50)
val = []
for train_data, test_data in skf.split(X, y):
    X_train, X_test = X[train_data], X[test_data]
    y_train, y_test = y[train_data], y[test_data]

    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    test = np.where(predict == y_test, 1, 0)
    test = np.sum(test) / len(y_test)
    val.append(test)

avg = np.sum(val) / len(val)
print('AdaBoost scores: ', val, '(avg = ', avg, ')\n')


clf = GradientBoostingClassifier(random_state=RANDOM_STATE, learning_rate=1, subsample=0.5, n_estimators=50, min_samples_leaf=3)
val = []
for train_data, test_data in skf.split(X, y):
    X_train, X_test = X[train_data], X[test_data]
    y_train, y_test = y[train_data], y[test_data]

    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    test = np.where(predict == y_test, 1, 0)
    test = np.sum(test) / len(y_test)
    val.append(test)

avg = np.sum(val) / len(val)
print('Gradient boosting scores: ', val, '(avg = ', avg, ')\n')


# Plot OOB estimates for Gradient Boosting Classifier.
clf = GradientBoostingClassifier(random_state=1, n_estimators=50, learning_rate=1.0, subsample=0.5, max_depth=1, min_samples_leaf=3)
clf.fit(X, y)
cumsum = np.cumsum(clf.oob_improvement_)
plt.figure()
plt.plot(range(0, len(cumsum)), cumsum)
plt.xlabel('Iteration')
plt.ylabel('OOB cumulated')
plt.show()

