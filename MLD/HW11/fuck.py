import random
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math
import time

trainFile = open("converted.txt")
trainLines = trainFile.readlines()

trainingData = list()
X = list()
Y = list()
for i in trainLines:
	data = i.split()
	# print(data)
	X.append([float(data[1]),float(data[2])])
	if (int(data[0]) == 1):
		Y.append(1)
		plt.plot(X[len(X)-1][0],X[len(X)-1][1],'ob')
	else:
		Y.append(-1)
		plt.plot(X[len(X)-1][0],X[len(X)-1][1],'xr')


plt.show()