"""
========================
Plotting Learning Curves
========================

"""
print(__doc__)

import numpy as np
import formatData as fD
import pandas as pd
import matplotlib.pyplot as plt
# import the different classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import collections

def plot_learning_curve(estimators, estimator_names, title, X, y, ylim=None, cv=None,

                        n_jobs=1, ts=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimators : array of classifiers that implement the "fit" and "predict" methods

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    est_len = len(estimators)

    train_sizes = [0] * est_len
    train_scores = [0]* est_len
    test_scores = [0]* est_len

    train_scores_mean = [0] * est_len
    train_scores_std = [0] * est_len
    test_scores_mean = [0] * est_len
    test_scores_std = [0] * est_len

    # allEstimatorInfo = {}
    # for i in range(len(estimators)):
    #     allEstimatorInfo[esimators[i]] = [train_sizes[i],train_scores[i],test_scores[i]

    for i in range(est_len): 
        train_sizes[i], train_scores[i], test_scores[i] = learning_curve(
        estimators[i], X, y, cv=cv, n_jobs=n_jobs, train_sizes=ts)

        train_scores_mean[i] = np.mean(train_scores[i], axis=1)
        train_scores_std[i] = np.std(train_scores[i], axis=1)
        test_scores_mean[i] = np.mean(test_scores[i], axis=1)
        test_scores_std[i] = np.std(test_scores[i], axis=1)


    plt.grid()


    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]

    for j in range(est_len):

        # plt.fill_between(train_sizes[j], train_scores_mean[j] - train_scores_std[j],
        #                 train_scores_mean[j] + train_scores_std[j], alpha=0.1,
        #                 color=colors[j])
        plt.fill_between(train_sizes[j], test_scores_mean[j] - test_scores_std[j],
                    test_scores_mean[j] + test_scores_std[j], alpha=0.1, color=colors[j])


        # plt.plot(train_sizes[j], train_scores_mean[j], 'o-', color="r",
        #         label="Training score")
        plt.plot(train_sizes[j], test_scores_mean[j], 'o-', color=colors[j],

                label=estimator_names[j])
    
    plt.legend(loc="best")
    
    return plt



#ylim is min and max response times
#random forest
#lin reg
#SVC
#adaboost
#nearest neighbor
#neural nets

# data = project_partition.partition_data()
# X = data['validation']
# y = data['test']


X,y = fD.csvToArray("data/train.csv")
y = fD.biny(y)
maximum = max(y)
minimum = min(y)
# class data_set:

#     def __init__(self):
#         header = None
#         data = None
#         target = None

# # import some data to play with
# dataset = pd.read_csv('data/train.csv')
# data = data_set()
# data.header = dataset.columns
# print data.header
# data.data = dataset[data.header[:17]]

# for col in data.header:
#     data.data = data.data[~data.data[col].isin(['?'])]

# data.target = data.data['INCIDENT_RESPONSE_SECONDS_QY']

# data.data.pop('INCIDENT_DATETIME')
# data.data.pop('INCIDENT_RESPONSE_SECONDS_QY')
# # print data.data
# X = data.data
# y = data.target
# X,y = fD.csvToArray("/Users/morganwalker/Desktop/Spring 2017/Machine Learning/ny-ems-response-predictor/data/train.csv")
# y = fD.biny(y)
# maximum = max(y)
# minimum = min(y)
X,y = fD.getData("data/train.csv")

# print(X)
# print(y)

# data = load_digits()
# X=data.data
# print("hello")
# print(type(X))
# # y=data.target
# print(y)

title = "Learning Curves"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
#cv = 10

estimators = []
estimators.append(RandomForestClassifier())
estimators.append(GaussianNB())
estimators.append(SVC(gamma=0.001))
estimators.append(MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1))
estimators.append(LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1))
estimators.append(AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None))
estimator_names = ["Random Forest","Gaussian Naive Bayes","Support Vector Machine","Multi-Layer Perceptron","Linear Regression","AdaBoost"]
# plot_learning_curve(estimators, title, X, y, ylim=(0.7, 1.01), cv=10,  n_jobs=1)
plot_learning_curve(estimators, estimator_names, title, X, y, cv=10,  n_jobs=1)

# title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# # SVC is more expensive so we do a lower number of CV iterations:
# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# estimator = SVC(gamma=0.001)
# plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

plt.show()
