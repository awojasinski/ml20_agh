import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os

path = os.getcwd() + '/dane_egz.txt'
data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])

print(data.describe())

X = np.array([data['Exam1'], data['Exam2']],).T
X = np.append(np.full(shape=(X.shape[0], 1), fill_value=1), X, axis=1)
y = np.array([data['Admitted']]).T

print(X.shape)
print(y.shape)