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

def gauss(z):
	return 2.718281828459**(-0.5*z**2)

def calc(Z,w,Y):
	Zw = np.dot(Z,w)
	# print(Zw)
	ein = 0
	found = False
	start = random.randrange(0,len(Y))
	for i in range(start,len(Y)):
		if (not ((Y[i] > 0 and Zw[i] > 0) or (Y[i] < 0 and Zw[i] < 0))):
			ein += 1
			if (not found):
				found = True
				w = w + Y[i]*Z[i]
	for i in range(start):
		if (not ((Y[i] > 0 and Zw[i] > 0) or (Y[i] < 0 and Zw[i] < 0))):
			ein += 1
			if (not found):
				found = True
				w = w + Y[i]*Z[i]
	# print(ein)
	ein /= len(Y)
	return (w,ein)


def RBF(X,Y,current,val,k):
	centers = [random.randrange(0,len(X))]
	for i in range(k-1):
		centers.append(findFurthest(centers,X))
	# print(centers)
	regions = list()
	for i in range(len(centers)):
		regions.append(list())
	for i in range(len(X)):
		regions[closestCenter(centers,X[i],X)].append(i)
	# print(regions)

	middle = list()
	for i in range(len(regions)):
		newmiddle = [0,0]
		for j in range(len(regions[i])):
			newmiddle[0] += X[regions[i][j]][0]
			newmiddle[1] += X[regions[i][j]][1]
		newmiddle[0] /= len(regions[i])
		newmiddle[1] /= len(regions[i])
		middle.append(newmiddle)

	r = 2/(k**0.5)
	Z = list()
	for i in range(len(X)):
		z = [1]
		for j in range(len(middle)):
			z.append(gauss(distance(X[i],middle[j])/r))
		Z.append(z)
	Z = np.array(Z)
	w = np.array([0]*(k+1))
	bestw = w
	Ein = -1
	for i in range(1000):
		(w,newEin) = calc(Z,w,Y)
		if (newEin < Ein or Ein == -1):
			bestw = w
			Ein = newEin

	# print(Ein)
	curz = [1]
	for i in range(len(middle)):
		curz.append(gauss(distance(current,middle[j])/r))
	if ((np.dot(curz,w) < 0 and val < 0) or (np.dot(curz,w) > 0 and val > 0)):
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
	# print(data)
	X.append([float(data[1]),float(data[2])])
	if (int(data[0]) == 1):
		Y.append(1)
	else:
		Y.append(-1)


Ecv = list()
k = list()

for i in range(1,100,2):
	print(i)
	e = 0
	for j in range(0,300):
		print(j)
		current = X.pop(0)
		val = Y.pop(0)
		# print(current,val)
		e += RBF(X,Y,current,val,i)
		# break
		# print(e)
		X.append(current)
		Y.append(val)
	e /= 300
	Ecv.append(e)
	k.append(i)
	# break

for i in range(101,200,2):
	print(i)
	e = 0
	for j in range(0,300,10):
		print(j)
		current = X.pop(0)
		val = Y.pop(0)
		# print(current,val)
		e += RBF(X,Y,current,val,i)
		# break
		# print(e)
		X.append(current)
		Y.append(val)
	e /= 300/10
	Ecv.append(e)
	k.append(i)

for i in range(201,300,2):
	print(i)
	e = 0
	for j in range(0,300,20):
		print(j)
		current = X.pop(0)
		val = Y.pop(0)
		# print(current,val)
		e += RBF(X,Y,current,val,i)
		# break
		# print(e)
		X.append(current)
		Y.append(val)
	e /= 300/20
	Ecv.append(e)
	k.append(i)

plt.title("E_CV for various k for RBF Rule")
plt.xlabel("k")
plt.ylabel("E_CV")
plt.plot(k,Ecv)
plt.show()
 



