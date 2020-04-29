import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import datasets

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Standaryzacja danych
np.where(X == float('Inf'), float('NaN'), X)
X = X[~np.isnan(X).any(axis=1)]
X_std = scale(X)

# Podział na zbiór treningowy i testowy(75%-25%)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.25)

# Utworzenie macierzy kowariancji
covMat = np.cov(X_std.T)
print('Covariance matrix: \n', covMat)

# Wyznaczenie wektorów własnych macierzy i wartości własnych
eig_vals, eig_vecs = np.linalg.eig(covMat)
print('\nEigenvalues: \n', eig_vals)
print('\nEigenvectors: \n', eig_vecs)

# Macierz korelacji
corMat = np.corrcoef(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(corMat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# Macierz korelacji surowe dane
corMat2 = np.corrcoef(X.T)

eig_vals, eig_vecs = np.linalg.eig(corMat2)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# SVD
u, s, v = np.linalg.svd(X_std.T)
print(u)

#  Wybór kluczowych komponentów

for ev in eig_vecs.T:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('\nEverything ok!\n')


# Utworzenie listy wartości własnych i wektorów własnych
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

# Posortowanie listy
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Wyświetlenie posortowanej listy
print('\nEigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

# Wariancja wyjaśniona
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(10), var_exp, alpha=0.5, align='center', label='individual explained variance')
    plt.plot(var_exp, '.-m')
    plt.step(range(10), cum_var_exp, where='mid', label='cumulative explained variance')
    plt.ylabel('Wartosci wspolczynnikow wariancji wyjasnionej')
    plt.xlabel('Numery wartosci wlasnych')
    plt.legend(loc='best')
    plt.tight_layout()

plt.show()

# Wybór k największych wektorów
for k, ev in enumerate(cum_var_exp):
    if ev >= 95:
        print('\n', k, ' największych wektorów własnych zawiera ', round(ev, 2), '% informacji')
        print('pozostałe ', len(cum_var_exp)-k, ' składowe można pominąć')
        break
