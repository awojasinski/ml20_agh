import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model as linm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

dataset = pd.read_csv('boston.csv')

X = dataset.drop('MEDV', axis=1)
y = dataset['MEDV']

# Podział na zbiór uczący i testowy (70%, 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Regresja liniowa (podstawowy model)
model = linm.LinearRegression()
model.fit(X_train, y_train)
print('Regresja liniowa')
print('Skutecznosc w zbiorze treningowym: {}'.format(model.score(X_train, y_train)))
print('Skutecznosc w zbiorze testowym: {}'.format(model.score(X_test, y_test)))

# Aproksymacja wielomianowa
scaler = StandardScaler()
X_2 = scaler.fit_transform(X)
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y, test_size=0.3)

steps = [
    ('poly', PolynomialFeatures(degree=2)),
    ('model', linm.LinearRegression())
]
pipe = Pipeline(steps)
pipe.fit(X_2_train, y_2_train)

print('\nRegresja liniowa (aproksymacja liniowa)')
print('Blad treningowy: ', pipe.score(X_2_train, y_2_train))
print('Blad testowy: ', pipe.score(X_2_test, y_2_test))

# Regularyzacja metodą Ridge
steps = [
    ('poly', PolynomialFeatures(degree=2)),
    ('model', linm.Ridge(alpha=10))
]
pipe = Pipeline(steps)
pipe.fit(X_2_train, y_2_train)
print('\n ridge')
print('Blad treningowy: ', pipe.score(X_2_train, y_2_train))
print('Blad testowy: ', pipe.score(X_2_test, y_2_test))

# Zależność pomiędy skutecznością regularyzacj metodą Ridge a parametrem alpha
test_score = []
train_score = []
alpha_arr = np.linspace(0.001, 500, 1000)
for i in alpha_arr:
    steps = [
        ('poly', PolynomialFeatures(degree=2)),
        ('model', linm.Ridge(alpha=i))
    ]
    pipe = Pipeline(steps)
    pipe.fit(X_2_train, y_2_train)
    train_score.append(pipe.score(X_2_train, y_2_train))
    test_score.append(pipe.score(X_2_test, y_2_test))

plt.plot(alpha_arr, train_score, alpha_arr, test_score)
plt.grid(True)
plt.xlabel('alpha')
plt.ylabel('score')
plt.legend(['Train', 'Test'])
plt.title('Skuteczność od parametru alpha w regularyzacji metodą Ridge')
plt.show()

# Regularyzacja metodą Lasso
alpha_arr = np.linspace(0, 1, 1000)
max_score = 0
max_alpha = 0
for i in alpha_arr:
    steps = [
        ('poly', PolynomialFeatures(degree=2)),
        ('model', linm.Lasso(alpha=i))
    ]
    pipe = Pipeline(steps)
    pipe.fit(X_2_train, y_2_train)
    if max_score < pipe.score(X_2_test, y_2_test):
        max_score = pipe.score(X_2_test, y_2_test)
        max_alpha = i

steps = [
    ('poly', PolynomialFeatures(degree=2)),
    ('model', linm.Lasso(alpha=max_alpha))
]
pipe = Pipeline(steps)
pipe.fit(X_2_train, y_2_train)
print('\nRegresja Lasso')
print('Alpha = ', max_alpha)
print('Blad treningowy: ', pipe.score(X_2_train, y_2_train))
print('Blad testowy: ', pipe.score(X_2_test, y_2_test))
