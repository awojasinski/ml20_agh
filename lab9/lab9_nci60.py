import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def printInfo_dataset(df):
    print('\n First 5 elements of the dataset')
    print(df.head())
    print('\nInformations about the columns')
    print(df.info())
    print('\nColumns means of the data')
    print(df.mean())
    print('\nVariances of the variables')
    print(df.var())
    pass


df = pd.read_csv('nci60.csv').drop('Unnamed: 0', axis=1)
df.columns = np.arange(df.columns.size)
y = pd.read_csv('nci60_y.csv', usecols=[1], skiprows=1, names=['type'])

print('---------------------------------------------')
print('DATA BEFORE PREPROCESSING')
printInfo_dataset(df)
print('---------------------------------------------')

X = pd.DataFrame(scale(df))
print('---------------------------------------------')
print('DATA AFTER PREPROCESSING')
printInfo_dataset(X)
print('\nShape')
print(X.shape)
print('---------------------------------------------')

pca = PCA()
df_plot = pd.DataFrame(pca.fit_transform(X))

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,6))
color_idx = pd.factorize(y.type)[0]
cmap = mpl.cm.hsv

# Left plot
ax1.scatter(df_plot.iloc[:,0], df_plot.iloc[:,1], c=color_idx, cmap=cmap, alpha=0.5, s=50)
ax1.set_ylabel('Principal Component 2')
ax1.grid(True)

# Right plot
ax2.scatter(df_plot.iloc[:,0], df_plot.iloc[:,2], c=color_idx, cmap=cmap, alpha=0.5, s=50)
ax2.set_ylabel('Principal Component 3')
ax2.grid(True)

# Custom legend for the classes (y) since we do not create scatter plots per class (which could have their own labels).
handles = []
labels = pd.factorize(y.type.unique())
norm = mpl.colors.Normalize(vmin=0.0, vmax=14.0)
for i, v in zip(labels[0], labels[1]):
    handles.append(mpl.patches.Patch(color=cmap(norm(i)), label=v, alpha=0.5))
ax2.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# xlabel for both plots
for ax in fig.axes:
    ax.set_xlabel('Principal Component 1')

fig.show()

pd.DataFrame([df_plot.iloc[:, :5].std(axis=0, ddof=0), pca.explained_variance_ratio_[:5],
              np.cumsum(pca.explained_variance_ratio_[:5])],
             index=['Standard Deviation', 'Proportion of Variance', 'Cumulative Proportion'],
             columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

plt.figure()
df_plot.iloc[:, :10].var(axis=0, ddof=0).plot(kind='bar', rot=0)
plt.ylabel('Variances')
plt.xlabel('Principal components')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Left plot
ax1.plot(pca.explained_variance_ratio_, '-o')
ax1.set_ylabel('Proportion of Variance Explained')
ax1.set_ylim(ymin=-0.01)
ax1.grid(True)

# Right plot
ax2.plot(np.cumsum(pca.explained_variance_ratio_), '-ro')
ax2.set_ylabel('Cumulative Proportion of Variance Explained')
ax2.set_ylim(ymax=1.05)
ax2.grid(True)
for ax in fig.axes:
    ax.set_xlabel('Principal Component')
    ax.set_xlim(-1, 65)

fig.show()
