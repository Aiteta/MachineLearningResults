import random
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math
import time

def distance(x1,x2):
	return ((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)**0.5

def findFurthest(centers,X):
	furthest = -1
	dist = 0
	for i in range(len(X)):
		newDist = distance(X[i],X[centers[closestCenter(centers,X[i],X)]])
		if (newDist > dist):
			furthest = i
			dist = newDist

	return furthest

def closestCenter(centers,x,X):
	closest = -1
	dist = 0
	for i in range(len(centers)):
		newDist = distance(x,X[centers[i]])
		if (closest == -1 or newDist < dist):
			dist = newDist
			closest = i
	return closest

def findBrute(X,x):
	closest = -1
	dist = 0
	for i in range(len(X)):
		newDist = distance(x,X[i])
		if (closest == -1 or newDist < dist):
			dist = newDist
			closest = i
	return closest

def findOpt(X,x,regions,middles,radii):
	possibleRegions = list()
	for i in range(10):
		dist1a = distance(middles[i],x)+radii[i]
		dist1b = distance(middles[i],x)-radii[i]
		add = True
		filter(lambda j: dist1a > distance(middles[j],x)-radii[j], possibleRegions)
		for j in possibleRegions:
			if (distance(middles[j],x)+radii[j] < dist1b):
				add = False
				break
		if (add):
			possibleRegions.append(i)
	# print(len(possibleRegions))
	closest = -1
	dist = 0
	for i in range(len(possibleRegions)):
		for j in range(len(regions[possibleRegions[i]])):
			newDist = distance(x,X[regions[possibleRegions[i]][j]])
			if (closest == -1 or newDist < dist):
				dist = newDist
				closest = regions[possibleRegions[i]][j]
	return closest

def findKClosest(X,Y,current,k):
	closest = list()
	maxlen = 0
	num = 0
	for i in range(len(X)):
		if (len(closest) < k):
			closest.append(i)
			dist = distance(X[i],current)
			if (maxlen < dist):
				maxlen = dist
				num = len(closest)-1
		elif (distance(X[i],current) < maxlen):
			closest.pop(num)
			closest.append(i)
			maxlen = 0
			num = 0
			for j in range(len(closest)):
				dist = distance(X[closest[j]],current)
				if (maxlen < dist):
					maxlen = dist
					num = j
	total = 0
	# print(len(closest), k)
	for i in range(len(closest)):
		total += Y[closest[i]]
	if (total > 0):
		return 1
	else:
		return -1

trainFile = open("train.txt")
trainLines = trainFile.readlines()

X = list()
Y = list()
for i in trainLines:
	data = i.split()
	# print(data)
	X.append([float(data[1]),float(data[2])])
	if (int(data[0]) == 1):
		Y.append(1)
	else:
		Y.append(-1)



# Ein = 0
# for i in range(len(X)):
# 	if (findKClosest(X,Y,X[i],11) != Y[i]):
# 		Ein += 1
# 	if (Y[i] < 0):
# 		plt.plot(X[i][0],X[i][1],'xr')
# 	else:
# 		plt.plot(X[i][0],X[i][1],'ob')

testFile = open("test.txt")
testLines = testFile.readlines()

Etest = 0

X1 = list()
Y1 = list()
n = 0
for i in testLines:
	print(n)
	n += 1
	data = i.split()
	# print(data)
	x = [float(data[1]),float(data[2])]
	y = 0
	if (int(data[0]) == 1):
		y = 1
		plt.plot(x[0],x[1],'ob')
	else:
		y = -1
		plt.plot(x[0],x[1],'xr')
	if (findKClosest(X,Y,x,11) != y):
		Etest += 1

print(Etest/n)

for i in range(51):
	for j in range(51):
		x = [i/25-1,j/25-1]
		if (findKClosest(X,Y,x,11) < 0):
			plt.plot(x[0],x[1],'sr', alpha = 0.2)
		else:
			plt.plot(x[0],x[1],'sb', alpha = 0.2)





plt.title("Decision Boundary with Test Data (k-NN)")
plt.xlabel("Average Intensity")
plt.ylabel("Vertical-Horizontal Ratio")
plt.show()
 








