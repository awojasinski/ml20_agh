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

# Load dataset
df = pd.read_csv('usarrests.csv', index_col=0)

print('---------------------------------------------')
print('DATA BEFORE PREPROCESSING')
printInfo_dataset(df)
print('---------------------------------------------')

X = pd.DataFrame(scale(df), index=df.index, columns=df.columns)

print('---------------------------------------------')
print('DATA AFTER PREPROCESSING')
printInfo_dataset(X)
print('---------------------------------------------')

pca_loadings = pd.DataFrame(PCA().fit(X).components_.T, index=df.columns, columns=['V1', 'V2', 'V3', 'V4'])
print(pca_loadings)

pca = PCA()
df_plot = pd.DataFrame(pca.fit_transform(X), columns=['PC1', 'PC2', 'PC3', 'PC4'], index=X.index)
print(df_plot.head())

fig, ax1 = plt.subplots(figsize=(9, 7))
ax1.set_xlim(-3.5, 3.5)
ax1.set_ylim(-3.5, 3.5)

# Plot Principal Components 1 and 2
for i in df_plot.index:
    ax1.annotate(i, (-df_plot.PC1.loc[i], -df_plot.PC2.loc[i]), ha='center')

# Plot reference lines
ax1.hlines(0, -3.5, 3.5, linestyles='dotted', colors='grey')
ax1.vlines(0, -3.5, 3.5, linestyles='dotted', colors='grey')
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')

# Plot Principal Component loading vectors, using a second y-axis.
ax2 = ax1.twinx().twiny()
ax2.set_ylim(-1, 1)
ax2.set_xlim(-1, 1)
ax2.set_xlabel('Principal Component loading vectors', color='red')

# Plot labels for vectors. Variable 'a' is a small offset parameter to separate arrow tip and text.
a = 1.07
for i in pca_loadings[['V1', 'V2']].index:
    ax2.annotate(i, (-pca_loadings.V1.loc[i]*a, -pca_loadings.V2.loc[i]*a), color='red')

# Plot vectors
ax2.arrow(0, 0, -pca_loadings.V1[0], -pca_loadings.V2[0])
ax2.arrow(0, 0, -pca_loadings.V1[1], -pca_loadings.V2[1])
ax2.arrow(0, 0, -pca_loadings.V1[2], -pca_loadings.V2[2])
ax2.arrow(0, 0, -pca_loadings.V1[3], -pca_loadings.V2[3])

fig.show()

print('\nExplained variance: ', pca.explained_variance_)
print('\nExplained variance ratio: ', pca.explained_variance_ratio_)

plt.figure(figsize=(7, 5))
plt.plot([1, 2, 3, 4], pca.explained_variance_ratio_, '-o')
plt.ylabel('Proportion of Variance Explained')
plt.xlabel('Principal Component')
plt.xlim(0.75, 4.25)
plt.ylim(0, 1.05)
plt.xticks([1, 2, 3, 4])
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot([1, 2, 3, 4], np.cumsum(pca.explained_variance_ratio_), '-s')
plt.ylabel('Proportion of Variance Explained')
plt.xlabel('Principal Component')
plt.xlim(0.75, 4.25)
plt.ylim(0, 1.05)
plt.xticks([1, 2, 3, 4])
plt.grid(True)
plt.show()
