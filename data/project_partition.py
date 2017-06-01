import random
import csv
import json
import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pprint as pp

def standardize_data(data):
	header = data.columns

	col_values = {}

	enc_data = data.copy()
	# get unique values
	for col in header:
		enc_data[col].replace('Y', 1, inplace=True)
		enc_data[col].replace('N', 0, inplace=True)

		if (data[col].dtype != np.float64 and data[col].dtype != np.int64 and col !='INCIDENT_DATETIME'):
			unique = data[col].unique()
			encodings = {}
			for i, val in enumerate(unique):
				encodings[val] = i
				enc_data[col].replace(val, i, inplace=True)

			col_values[col] = encodings

		print col

	with open("encodings.json","w") as f:
	    json.dump(col_values,f)

	enc_data.to_csv('encoded_data.csv')

def standardize_dataset(data, name):
	header = data.columns

	col_values = {}

	enc_data = data.copy()

	# get unique values
	for col in header:

		print 'starting ' + col
		enc_data[col].replace('Y', 1, inplace=True)
		enc_data[col].replace('N', 0, inplace=True)

		if (data[col].dtype != np.float64 and data[col].dtype != np.int64 and col !='INCIDENT_DATETIME'):
			unique = data[col].unique()
			encodings = {}
			for i, val in enumerate(unique):
				encodings[val] = i
				enc_data[col].replace(val, i, inplace=True)

			col_values[col] = encodings

		print 'finishing ' + col

	with open('encodings_'+ name + '.json',"w") as f:
	    json.dump(col_values,f)

	enc_data.to_csv('encoded_data_'+ name + '.csv')

def partition_data(file):

	data = pd.read_csv(file)
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

	return {'header':header,'train':train,'test':test, 'validation':validation}

file1 = 'EECS349_formatted.csv'
file2 = '../encoded_data_test.csv'

# data = pd.read_csv(file1)
data = partition_data(file2)
# partitions = ['train.csv', 'test.csv', 'validation.csv']

# train = pd.read_csv('train.csv')
# standardize_dataset(train, 'train')

# test = pd.read_csv('test.csv')
# standardize_dataset(test, 'test')

# validation = pd.read_csv('validation.csv')
# standardize_dataset(validation, 'validation')

# standardize_data(data)
