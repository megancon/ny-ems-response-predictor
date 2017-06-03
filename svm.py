print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import formatData as fD
import project_partition
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

class data_set:

	def __init__(self):
		header = None
		data = None
		target = None

# import some data to play with
dataset = pd.read_csv('data/train.csv')
data = data_set()
data.header = dataset.columns
data.data = dataset[data.header[:17]]

data.target = data.data['INCIDENT_RESPONSE_SECONDS_QY']

for col in data.header:
	data.data = data.data[~data.data[col].isin(['?'])]

data.data.pop('INCIDENT_DATETIME')
data.data.pop('INCIDENT_RESPONSE_SECONDS_QY')
# print data.data
X = data.data
y = data.target

print "good"

h = .1  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
# svc = svm.SVC(kernel='linear', C=C).fit(X, y)

# rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
# poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)
print "svm complete"
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
										 np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
					'LinearSVC (linear kernel)',
					'SVC with RBF kernel',
					'SVC with polynomial (degree 3) kernel']
print "preloop"

# for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
for i, clf in enumerate((lin_svc)):
		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, x_max]x[y_min, y_max].
		plt.subplot(2, 2, i + 1)
		plt.subplots_adjust(wspace=0.4, hspace=0.4)

		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

		# Put the result into a color plot
		Z = Z.reshape(xx.shape)
		plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

		# Plot also the training points
		plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
		plt.xlabel('Sepal length')
		plt.ylabel('Sepal width')
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.xticks(())
		plt.yticks(())
		plt.title(titles[i])
		print "hi"

print "hello"

plt.show()