import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve

def linReg(features,labels):
	lr = LinearRegression()
	lr.fit(features,labels)
	print("coef: " + str(lr.coef_))
	print("intercept: " + str(lr.intercept_))
	plt.scatter(features,labels)
	plot.show


digits = load_digits()
features, labels = digits.data, digits.target

linReg(features,labels)