import random
import numpy as np

def Pre_Processing_Data(path, My_data):
	data = []
	with open(path, 'r') as f:
		for line in f:
			line = line.split('\n')
			new = line[0].split(" ")
			if '1st' in new:
				social_class = 0
			elif '2nd' in new:
				social_class = 1
			elif '3rd' in new:
				social_class = 2
			elif 'crew' in new:
				social_class = 3

			if 'adult' in new:
				age = 0
			elif 'child' in new:
				age = 1

			if 'male' in new:
				sex = 0
			elif 'female' in new:
				sex = 1

			if 'yes' in new:
				isAlive = 1
			elif 'no' in new:
				isAlive = 0

			My_data.write(str(social_class) + " " + str(age) + " " + str(sex) + " " + str(isAlive) + "\n")

def Shuffle_Data(path, My_data):
	data = []
	with open(path, 'r') as f:
		for line in f:
			line = line.split('\n')
			line = line[0].split(" ")
			data.append([int(line[0]), int(line[1]), int(line[2]), int(line[3])])
	random.shuffle(data)
	for i in data:
		My_data.write(str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + " " + str(i[3]) + "\n")
		
def SplitData(My_data):
	data = np.array(My_data)
	X = data[:, 0:3]
	y = data[:, 3]

	# Chia dữ liệu train và test
	a = len(X)//3 # tỉ lệ train : test  = 2:1
	X_train = X[0:2*a]
	y_train = y[0:2*a]

	X_test = X[2*a:]
	y_test = y[2*a:]
	return X_train, y_train, X_test, y_test

def Load_Data(path):
	data = []
	with open(path, 'r') as f:
		for line in f:
			line = line.split('\n')
			line = line[0].split(" ")
			data.append([int(line[0]), int(line[1]), int(line[2]), int(line[3])])

	return SplitData(data)





