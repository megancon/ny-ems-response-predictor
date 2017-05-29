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
train = data.sample(frac=0.6, random_state=1)
train.to_csv('train.csv')

#gets the left out portion of the dataset
leftover = data.loc[~data.index.isin(train.index)]

test = leftover.sample(frac=0.5, random_state=1)
test.to_csv('test.csv')
validation = leftover.loc[~leftover.index.isin(test.index)]
validation.to_csv('validation.csv')
col_values = {}
# get unique values
for col in header:
	col_values[col] = train[col].unique()

pp.pprint(col_values)
# enc = preprocessing.OneHotEncoder()
# enc.fit(train)  


# answer = train["INCIDENT_RESPONSE_SECONDS_QY"]
# print answer

# train.drop("INCIDENT_RESPONSE_SECONDS_QY")
# f = open("../data/EECS349_formatted.csv", 'rt')

# reader = csv.reader(f)
# data = []
# for row in reader:
#     data.append(row)
# f.close()

# random.shuffle(data[1:])

# train = int(0.6*len(data[1:]))
# test = int(0.2*len(data[1:]))
# validation = int(0.2*len(data[1:]))



# train_data = data[:][:-1]
# output = train_data[:][-1]
# print train_data

# test_data = np.asarray(data[train:train+test+1])
# validation_data = np.asarray(data[train+test+1:train+test+validation+1])

