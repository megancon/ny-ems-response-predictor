import random
import csv
import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pprint as pp
import json


data = pd.read_csv('train.csv')
header = data.columns

col_values = {}

le = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()

# get unique values
# for col in header:
# 	if (data[col].dtype != np.float64 and data[col].dtype != np.int64):
# 		unique = data[col].unique()
# 		le.fit_transform(data[col])
# 		# enc.fit_transform(data[col])
# 		encodings = []
# 		for val, i in enumerate(unique):
# 			encodings.append({val: i})

# 		col_values[col] = encodings

value_encodings = json.loads('value_encodings.json')
enc_data = data.copy()

enc_data.replace(value_encodings)

nn=NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(enc_data)



# enc_data.to_csv('encoded_data.csv')


# print col_values

# print data.head()