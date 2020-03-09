import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model as linm
from sklearn.model_selection import train_test_split

def normalize_data(X):
    X_mean = np.mean(X, axis=0)
    X_mean_M = np.full(X.shape, fill_value=X_mean)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean_M) / X_std
    return X_norm


boston = datasets.load_boston()

print(boston.DESCR)

boston_X = boston.data
boston_Y = boston.target
for col in range(boston_X.shape[1]):
    boston_X[:, col] = normalize_data(boston_X[:, col])

boston_Y = normalize_data(boston_Y)

# Podział na zbiór treningowy i testowy (70-30%)
X_train, X_test, Y_train, Y_test = train_test_split(boston_X, boston_Y, test_size=0.3)

# Stworzenie obiektu
reg_LinReg =linm.LinearRegression()
reg_Ridge = linm.Ridge(alpha=0.5)
reg_Lasso = linm.Lasso(alpha=5.1)
reg_ElNet =linm.ElasticNet(alpha=0.5, l1_ratio=0.5)

# Uczenie modelu przy pomocy bazy treningowej
reg_LinReg.fit(X_train, Y_train)
reg_Ridge.fit(X_train, Y_train)
reg_Lasso.fit(X_train, Y_train)
reg_ElNet.fit(X_train, Y_train)

# Przewidywanie wartości dla danych testowych
Y_LinReg_predicted = reg_LinReg.predict(X_test)
Y_Ridge_predicted = reg_Ridge.predict(X_test)
Y_Lasso_predicted = reg_Lasso.predict(X_test)
Y_ElNet_predicted = reg_ElNet.predict(X_test)

# Wyświetlenie parametrów prostej
print('\nCoefficients (Linear Regression): \n', reg_LinReg.coef_)
print('\nCoefficients (Ridge): \n', reg_Ridge.coef_)
print('\nCoefficients (Lasso): \n', reg_Lasso.coef_)
print('\nCoefficients (ElasticNet): \n', reg_ElNet.coef_)

#  Obliczamy rzeczywisty popełniony błąd średnio-kwadratowy
error_LinReg = np.mean((reg_LinReg.predict(X_test) - Y_test) ** 2)
error_Ridge = np.mean((reg_Ridge.predict(X_test) - Y_test) ** 2)
error_Lasso = np.mean((reg_Lasso.predict(X_test) - Y_test) ** 2)
error_ElNet = np.mean((reg_ElNet.predict(X_test) - Y_test) ** 2)
print("Residual sum of squares (Linear Regression): {}".format(error_LinReg))
print("Residual sum of squares (Ridge): {}".format(error_Ridge))
print("Residual sum of squares (Lasso): {}".format(error_Lasso))
print("Residual sum of squares (ElasticNet): {}".format(error_ElNet))

# Wyświetlenie prostych regresji obliczonych dla poszczególnych zmiennych
# wykonać tylko na testowym
theta0 = reg_LinReg.intercept_
ones = np.full(shape=Y_test.shape, fill_value=1)
for n, (theta1, feature_name) in enumerate(zip(reg_LinReg.coef_, boston['feature_names'])):
    theta = np.array([theta0, theta1])
    x = np.linspace(np.amin(X_test[:, n]), np.amax(X_test[:, n]), num=X_test.shape[0])
    X_data = np.array([ones, x]).T
    h = X_data @ theta.T
    plt.plot(X_test[:, n], Y_test, 'o')
    plt.plot(X_data[:, 1], h)
    plt.title('Prosta regresji')
    plt.xlabel(feature_name)
    plt.ylabel('Wartość domu [1000$]')
    plt.grid(True)
    plt.show()
