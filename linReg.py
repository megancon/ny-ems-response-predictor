import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
import formatData as fD

def linReg(features,labels):
	lr = LinearRegression()
	lr.fit(features,labels)
	print("coef: " + str(lr.coef_))
	print("intercept: " + str(lr.intercept_))
	plt.scatter(features,labels)
	plot.show

X, y = fD.csvToArray('data/train.csv')
features, labels = X, y

linReg(features,labels)