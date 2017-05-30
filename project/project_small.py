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


enc_data = data.copy()
# get unique values
for col in header:
	enc_data[col].replace('Y', 1, inplace=True)
	enc_data[col].replace('N', 0, inplace=True)

	# enc_data[col] = enc_data[col].convert_objects(convert_numeric=True)
	
	if (data[col].dtype != np.float64 and data[col].dtype != np.int64 and col !='INCIDENT_DATETIME'):
		unique = data[col].unique()
		# le.fit_transform(data[col])
		# enc.fit_transform(data[col])
		encodings = {}
		for i, val in enumerate(unique):
			# print val, i
			encodings[val] = i
			enc_data[col].replace(val, i, inplace=True)

		col_values[col] = encodings

with open("encodings_small.json","w") as f:
    json.dump(col_values,f)

enc_data.to_csv('encoded_data_small.csv')

# print col_values



# print data.head()