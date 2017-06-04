import numpy as np
import collections
import pandas as pd

BIN_SIZE = 666
# 1796
NUM_EXAMPLES = 20000
NUM_BINS = int(50) 

def csvToArray(path_to_file):
	X = []
	y = []
	data = open(path_to_file).readlines()
	row1 = data[0]
	headers = [x.rstrip() for x in row1.split(',')] 
	headers = headers [2:]
	zip_data = data[9]
	response_data
	i = 0
	for row in data:
		row = [x.rstrip() for x in row.split(',')]
		# remove the first two features because they are string values
		del row[:2]
		if '?' not in row:
			row = [int(x) for x in row]
			X.append(row[:-1])
			y.append(row[-1])
			# print(X)
			# print(y)
		i += 1
		if i > NUM_EXAMPLES:
			break
	X = [x for (Y,x) in sorted(zip(y,X))]
	y = sorted(y)
	return np.asarray(X),np.asarray(y),headers

def biny(y):
	new_y = []
	ave = 0
	# bin 10 items
	counter = 0
	for i in range(len(y)):
		ave += y[i]
		counter += 1
		if counter == BIN_SIZE:
			# calculate the average
			ave = ave/BIN_SIZE
			# update all of the past 10 elements to equal the average
			for j in range(BIN_SIZE):
				y[i-j] = ave
			# reset counter and average
			counter = 0
			ave = 0
	for j in range(BIN_SIZE):
		y[i-j] = ave
	return y

def biny2(y):
	new_y = []
	mini = int(min(y))
	maxi = int(max(y))
	range_stops = [i for i in range(mini, maxi, int(maxi/NUM_BINS))][1:]
	ranges = [[] for i in range(0, len(range_stops))]
	avgs = []

	for i in y:
		for j in range(len(ranges)):
			if i < range_stops[j]:
				ranges[j].append(i)
				break

	d = {}
	for i in range(len(range_stops)):
		d[range_stops[i]] = ranges[i]

	for vec in d.values():
		if len(vec) > 0:
			avgs.append(sum(vec)/len(vec))
		else:
			avgs.append(0)

	for i in range(len(y)):
		for j in range(len(ranges)):
			if y[i] < range_stops[j]:
				y[i] = avgs[j]
				break	
	return y

def shuffle_in_unison(a, b):
	rng_state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(rng_state)
	np.random.shuffle(b)
	return a,b

def getData(path_to_file):
	X,y,headers = csvToArray(path_to_file)
	y = biny(y)
	X,y = shuffle_in_unison(X,y)
	return X,y,headers

def arrayToCSV(path_to_file):
	features,labels,headers = getData(path_to_file)
	features = pd.DataFrame(features)
	features.columns = headers[:-1]
	features[headers[-1]] = labels
	features.to_csv('discreteEqualNum.csv', mode='a', index=False, header=True)

def reduceData(path_to_file):
	X = []
	y = []
	data = open(path_to_file).readlines()
	row1 = data[0]
	headers = [x.rstrip() for x in row1.split(',')] 
	headers = headers[2:]
	data = data[1:]
	i = 0
	for row in data:
		row = [x.rstrip() for x in row.split(',')]
		# remove the first two features because they are string values
		del row[:2]
		if '?' not in row:
			row = [int(x) for x in row]
			X.append(row[:-1])
			y.append(row[-1])
			# print(X)
			# print(y)
		i += 1
		if i > NUM_EXAMPLES:
			break
	features = pd.DataFrame(X)
	features.columns = headers[:-1]
	features[headers[-1]] = y
	features.to_csv('20000examples.csv', mode='a', index=False, header=True)

reduceData("/Users/morganwalker/Desktop/Spring 2017/Machine Learning/ny-ems-response-predictor/data/train.csv")
# def getTest


# a = np.array([[[  0.,   1.,   2.],
#                   [  3.,   4.,   5.]],

#                  [[  6.,   7.,   8.],
#                   [  9.,  10.,  11.]],

#                  [[ 12.,  13.,  14.],
#                   [ 15.,  16.,  17.]]])
# b = np.array([0.,
#                  3.,
#                  5.])
# print(a)
# print(b)
# print("shuffle")
# a,b = shuffle_in_unison_scary(a, b)
# print(a)
# print(b)
# X, y = csvToArray("/Users/morganwalker/Desktop/Spring 2017/Machine Learning/ny-ems-response-predictor/data/train.csv")
# y = biny(y)
# print(collections.Counter(y))


