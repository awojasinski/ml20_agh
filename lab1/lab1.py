import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

irisRaw = load_iris()

iris = pd.DataFrame(data= np.c_[irisRaw['data'], irisRaw['target']],
                    columns= irisRaw['feature_names'] + ['target'])

print(iris)
print()

print(iris.shape)
print()

for col in iris.columns:
    print(iris[col].describe())
    print()


for col in iris.columns:
    print(iris.groupby(col).describe())
    print()
    plot = iris.groupby(col).describe()
   

print(iris.head())
print()

new_iris = iris.dropna(axis=0, how='any')
if abs(len(new_iris) - len(iris)) > 0:
    print('Baza danych zawiera brakujące dane')
    print()
else:
    print('Baza danych nie zawiera brakujących danych')
    print()
    
print(iris.sort_values(by=iris.columns[1]))
print()

print('Index najmniejszej długosci płatka:')
print(iris[iris.columns[2]].idxmin())

print('\nIndex największej długosci płatka:')
print(iris[iris.columns[2]].idxmax())
print()

for col in iris.columns:
    print(col + '|std:')
    print(iris[col].std())
    print()

big_iris = iris['sepal length (cm)'] > iris['sepal length (cm)'].mean()

print(iris[big_iris])


fig, axes = plt.subplots(nrows= 2, ncols=2)
colors= ['blue', 'red', 'green']

for i, ax in enumerate(axes.flat):
    for label, color in zip(range(len(irisRaw.target_names)), colors):
        ax.hist(irisRaw.data[iris.target==label, i], label=             
                            irisRaw.target_names[label], color=color)
        ax.set_xlabel(irisRaw.feature_names[i])  
        ax.legend(loc='upper right')


plt.show()