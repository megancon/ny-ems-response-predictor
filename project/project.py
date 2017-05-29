import random
import csv
import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pprint as pp




data = pd.read_csv('train.csv')
header = data.columns

col_values = {}

le = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()

# get unique values
for col in header:
	col_values[col] = data[col].unique()
	le.fit_transform(data[col])
	# enc.fit_transform(data[col])



print data.head()
# nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(data)