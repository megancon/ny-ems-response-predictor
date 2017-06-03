import numpy as np
import collections

BIN_SIZE = 20
# 1796
NUM_EXAMPLES = 559001

def csvToArray(path_to_file):
	X = []
	y = []
	data = open(path_to_file).readlines()
	data = data[1:]
	i = 0
	for row in data:
		row = [x.rstrip() for x in row.split(',')]
		# remove the first two features because they are string values
		del row[:2]
		row = [0 if x == '?' else int(x) for x in row]
		X.append(row[:-1])
		y.append(row[-1])
		# print(X)
		# print(y)
		i += 1
		if i > NUM_EXAMPLES:
			break
	X = [x for (y,x) in sorted(zip(y,X))]
	y = sorted(y)
	return np.asarray(X),np.asarray(y)

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

def shuffle_in_unison(a, b):
	rng_state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(rng_state)
	np.random.shuffle(b)
	return a,b

def getData(path_to_file):
	X,y = csvToArray(path_to_file)
	y = biny(y)
	X,y = shuffle_in_unison(X,y)
	return X,y


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


