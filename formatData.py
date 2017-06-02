import numpy as np
import collections

BIN_SIZE = 20

def csvToArray(path_to_file):
    X = []
    y = []
    data = open(path_to_file).readlines()
    data = data[1:]
    i = 0
    for row in data:
        row = [x.rstrip() for x in row.split(',')]
        del row[:2]
        # row = [0 if x == '?' for x in row]
        row = [0 if x == '?' else int(x) for x in row]
        X.append(row[:-1])
        y.append(row[-1])
        # print(X)
        # print(y)
        i += 1
        if i > 1796:
            break
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



# X, y = csvToArray("/Users/morganwalker/Desktop/Spring 2017/Machine Learning/ny-ems-response-predictor/data/train.csv")
# y = biny(y)
# print(collections.Counter(y))


