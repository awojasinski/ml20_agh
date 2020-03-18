from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.model_selection import train_test_split
import pydot

iris = load_iris()

features_train, features_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size=0.3)

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

clf.predict(features_test)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf('iris.pdf')
