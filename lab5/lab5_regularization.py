import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('boston.csv')

X = dataset.drop('MEDV', axis=1)
y = dataset['MEDV']

# Podział na zbiór uczący i testowy (70%, 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression(solver='newton-cg', random_state=0).fit(X_train, y_train)

# model - model regress liniowej + fit
print('Blad treningowy: {}'.format(model.score(X_train, y_train)))
print('Blad testowy: {}'.format(model.score(X_test, y_test)))
