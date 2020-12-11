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

def findKClosest(X,Y,current,val,k):
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
	if ((val > 0 and total > 0) or (val < 0 and total < 0)):
		return 0
	else:
		return 1

trainFile = open("train.txt")
trainLines = trainFile.readlines()

trainingData = list()
X = list()
Y = list()
for i in trainLines:
	data = i.split()
	print(data)
	X.append([float(data[1]),float(data[2])])
	if (int(data[0]) == 1):
		Y.append(1)
	else:
		Y.append(-1)

k = list()
Ecv = list()

for i in range(1,300,2):
	print(i)
	e = 0
	for j in range(300):
		current = X.pop(0)
		val = Y.pop(0)
		# print(current,val)
		e += findKClosest(X,Y,current,val,i)
		# print(e)
		X.append(current)
		Y.append(val)
	e /= 300
	k.append(i)
	Ecv.append(e)

print(Ecv)


plt.title("E_CV for various k for k-NN Rule")
plt.xlabel("k")
plt.ylabel("E_CV")
plt.plot(k,Ecv)
plt.show()
 








