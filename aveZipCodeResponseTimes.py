import numpy as np
import collections
import pandas as pd

ZIP_CODE_INDEX = 9
RESPONSE_TIME_INDEX = 16
def calcAveResponseTimesPerZipCode(path_to_file):
	data = open(path_to_file).readlines()
	row1 = data[0]
	headers = [x.rstrip() for x in row1.split(',')] 
	headers = [headers[ZIP_CODE_INDEX],headers[RESPONSE_TIME_INDEX]]
	zipcodes = {}
	data = data[1:]
	for row in data:
		row = [x.rstrip() for x in row.split(',')]
		curr_zip = row[ZIP_CODE_INDEX]
		curr_response_time = int(row[RESPONSE_TIME_INDEX])
		if curr_zip in zipcodes.keys():
			zipcodes[curr_zip].append(curr_response_time)
		else:
			zipcodes[curr_zip] = [curr_response_time]
	ave_times_all_zips = []
	for curr_zip in zipcodes.keys():
		all_times = zipcodes[curr_zip]
		ave_time = np.mean(all_times)
		ave_times_all_zips.append([curr_zip,ave_time])
	# print(ave_times_all_zips)
	headers = ['ZIP_CODE','RESPONSE_TIME_SECONDS']
	df = pd.DataFrame(ave_times_all_zips,columns = headers)
	format = df.to_csv('zipCodes.csv', mode='a', index=False, header=True)


calcAveResponseTimesPerZipCode("/Users/morganwalker/Desktop/Spring 2017/Machine Learning/ny-ems-response-predictor/data/train.csv")