import random
import csv
import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pprint as pp



data = pd.read_csv('../data/EECS349_formatted.csv')
header = data.columns

#gets a random 60% of the entire set
train = data.sample(frac=0.001, random_state=1)
train.to_csv('trainsmall.csv')

#gets the left out portion of the dataset
# leftover = data.loc[~data.index.isin(train.index)]

# test = leftover.sample(frac=0.5, random_state=1)
# test.to_csv('test.csv')
# validation = leftover.loc[~leftover.index.isin(test.index)]
# validation.to_csv('validation.csv')

# enc = preprocessing.OneHotEncoder()
# enc.fit(train)  


