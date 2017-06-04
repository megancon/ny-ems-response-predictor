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
	lr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
	lr.fit(features,labels)
	print("coef: " + str(lr.coef_))
	print("intercept: " + str(lr.intercept_))
	print(len(features))
	print(len(labels))
	plt.scatter(features,labels)
	plot.show

def svmPlot(features,labels):
	model = SVC(gamma=0.001)
	model.fit(features,labels)
	model.predict()

# digits = load_digits()
# features, labels = digits.data, digits.target

# linReg(features,labels)

def main(path_to_file):
	# features, labels = fD.csvToArray(path_to_file)
	features, labels = fD.getData(path_to_file)
	linReg(features,labels)

main("/Users/morganwalker/Desktop/Spring 2017/Machine Learning/ny-ems-response-predictor/data/train.csv")