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
		newDist = distance(X[centers[closestCenter(centers,X[i],X)]],X[i])
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


def checkExists(centers,k):
	for i in range(len(centers)):
		if (centers[i] == k):
			return True
	return False

def LloydsInitial(centers,X):
	regions = list()
	for i in range(len(centers)):
		regions.append(list())
	for i in range(len(X)):
		regions[closestCenter(centers,X[i],X)].append(i)
	middle = list()
	for i in range(len(regions)):
		newmiddle = [0,0]
		for j in range(len(regions[i])):
			newmiddle[0] += X[regions[i][j]][0]
			newmiddle[1] += X[regions[i][j]][1]
		if (len(regions[i])):
			newmiddle[0] /= len(regions[i])
			newmiddle[1] /= len(regions[i])
		else:
			newmiddle[0] =  X[centers[i]][0]
			newmiddle[1] =  X[centers[i]][1]
		middle.append(newmiddle)
	return middle

def closestMiddle(middle,x):
	dist = -1
	out = 0
	# print(middle)
	for i in range(len(middle)):
		newDist = distance(middle[i],x)
		if (newDist < dist or dist == -1):
			dist = newDist
			out = i
	return out

def LLoyds(centers,X):
	middle = LloydsInitial(centers,X)
	for i in range(10):
		regions = list()
		for j in range(len(middle)):
			regions.append(list())
		for j in range(len(X)):
			regions[closestMiddle(middle,X[j])].append(j)
		# print(regions)
		middle.clear()
		for j in range(len(regions)):
			# print(len(regions[j]))
			newmiddle = [0,0]
			for k in range(len(regions[j])):
				newmiddle[0] += X[regions[j][k]][0]
				newmiddle[1] += X[regions[j][k]][1]
			newmiddle[0] /= len(regions[j])
			newmiddle[1] /= len(regions[j])
			middle.append(newmiddle)
		# print()
	return middle



def RBF(X,Y,k):
	# centers = list()
	# for i in range(k):
	# 	l = random.randrange(0,len(X))
	# 	while (checkExists(centers,l)):
	# 		l = random.randrange(0,len(X))
	# 	centers.append(l)
	centers = [random.randrange(0,len(X))]
	for i in range(k-1):
		centers.append(findFurthest(centers,X))
	# print(centers)

	middle = LLoyds(centers,X)
	# print(middle)
	# for i in range(len(middle)):
		# plt.plot(middle[i][0],middle[i][1],'kv')

	r = 0.8/(k**0.5)
	Z = list()
	for i in range(len(X)):
		z = [1]
		for j in range(len(middle)):
			# print(distance(X[i],middle[j])/r,gauss(distance(X[i],middle[j])/r))
			z.append(gauss(distance(X[i],middle[j])/r))
		Z.append(z)
	# print(Z)
	Z = np.array(Z)

	ZT = np.transpose(Z)
	ZTZ = np.dot(ZT,Z)
	ZTZ_1ZT = np.dot(np.linalg.inv(ZTZ),ZT)
	# ZTZ_1ZT = np.linalg.pinv(Z)
	w = np.dot(ZTZ_1ZT,np.array(Y))
	Ein = 0
	for i in range(len(X)):
		if (np.sign(np.dot(Z[i],w)) != Y[i]):
			Ein += 1
	Ein /= 300
	print(Ein)
	return (middle,w)

def createZ(middle,x):
	r = 0.8/(2**0.5)
	z = [1]
	for i in range(len(middle)):
		z.append(gauss(distance(x,middle[i])/r))
	return z





trainFile = open("train.txt")
trainLines = trainFile.readlines()

trainingData = list()
X = list()
Y = list()
for i in trainLines:
	data = i.split()
	# print(data)
	X.append([(float(data[1])-0.2)/0.68,(float(data[2])+0.2)/0.65])
	if (int(data[0]) == 1):
		Y.append(1)
		# plt.plot(float(data[1]),float(data[2]),'ob')
	else:
		Y.append(-1)
		# plt.plot(float(data[1]),float(data[2]),'xr')


(middle,w) = RBF(X,Y,9)

Ecv = list()
k = list()
for i in range(51):
	for j in range(51):
		x = [((i/25-1)-0.2)/0.68,((j/25-1)+0.2)/0.65]
		z = createZ(middle,x)
		if (np.dot(w,z) < 0):
			plt.plot(i/25-1,j/25-1,'sr', alpha = 0.2)
		else:
			plt.plot(i/25-1,j/25-1,'sb', alpha = 0.2)

testFile = open("test.txt")
testLines = testFile.readlines()

Etest = 0

X1 = list()
Y1 = list()
n = 0
Etest = 0
for i in testLines:
	print(n)
	n += 1
	data = i.split()
	# print(data)
	x = [(float(data[1])-0.2)/0.68,(float(data[2])+0.2)/0.65]
	y = 0
	if (int(data[0]) == 1):
		y = 1
		plt.plot(float(data[1]),float(data[2]),'ob')
	else:
		y = -1
		plt.plot(float(data[1]),float(data[2]),'xr')
	z = createZ(middle,x)
	yhat = np.dot(z,w)
	if (np.sign(yhat) != np.sign(y)):
		Etest += 1
print(Etest/n)





plt.title("Decision Boundary with Test Data (RBF)")
plt.xlabel("Average Intensity")
plt.ylabel("Vertical-Horizontal Ratio")
plt.show()
 

