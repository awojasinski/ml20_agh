import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

logreg = LogisticRegression(solver='newton-cg', random_state=0).fit(X_train, Y_train)

print('---------------------------------------------------------')
print('Prawdopodobieństwo przynależności obiektu do danej klasy:')
for n, prediction in enumerate(logreg.predict_proba(X_test)):
    p = prediction*100
    print('%d. Septal lenght: %.2f, Septal width: %.3f.' % (n+1, X_test[n, 0], X_test[n, 1]))
    print('\tProbability: 0<-%2.2f%s\t1<-%2.2f%s\t2<-%2.2f%s\n' % (p[0], '%', p[1], '%', p[2], '%'))
print('---------------------------------------------------------')
print('Algorithm accuracy score: ', logreg.score(X_test, Y_test))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.show()
