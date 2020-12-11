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



def RBF(X,Y,current,val,k):
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
	# print(w)

	# print(Ein)
	curz = [1]
	for i in range(len(middle)):
		curz.append(gauss(distance(current,middle[j])/r))

	yhat = np.dot(curz,w)
	# print(np.sign(yhat), np.sign(val))
	# for i in range(51):
	# 	for j in range(51):
	# 		zz = [1]
	# 		x = i/25-1
	# 		y = j/25-1
	# 		du = [x,y]
	# 		for k in range(len(middle)):
	# 			zz.append(gauss(distance(du,middle[k])/r))
	# 		if (np.dot(zz,w) < 0):
	# 			plt.plot(x,y,'sr', alpha = 0.1)
	# 		else:
	# 			plt.plot(x,y,'sb', alpha = 0.1)

	if ((yhat < 0 and val < 0) or (yhat > 0 and val > 0)):
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
	X.append([(float(data[1])-0.2)/0.68,(float(data[2])+0.2)/0.65])
	if (int(data[0]) == 1):
		Y.append(1)
		# plt.plot(X[len(X)-1][0],X[len(X)-1][1],'ob')
	else:
		Y.append(-1)
		# plt.plot(X[len(X)-1][0],X[len(X)-1][1],'xr')


Ecv = list()
k = list()

for i in range(1,50):
	print(i)
	e = 0
	for j in range(0,300,2):
		# print(j)
		current = X.pop(0)
		val = Y.pop(0)
		# print(current,val)
		e += RBF(X,Y,current,val,i)
		# break
		# print(e)
		X.append(current)
		Y.append(val)
	e /= 300/2
	print("\t",e)
	Ecv.append(e)
	k.append(i)
	# break

for i in range(50,100,10):
	print(i)
	e = 0
	for j in range(0,300,2):
		print(j)
		current = X.pop(0)
		val = Y.pop(0)
		# print(current,val)
		e += RBF(X,Y,current,val,i)
		# break
		# print(e)
		X.append(current)
		Y.append(val)
	e /= 300/2
	print("\t",e)
	Ecv.append(e)
	k.append(i)

for i in range(101,200,20):
	print(i)
	e = 0
	for j in range(0,300,10):
		# print(j)
		current = X.pop(0)
		val = Y.pop(0)
		# print(current,val)
		e += RBF(X,Y,current,val,i)
		# break
		# print(e)
		X.append(current)
		Y.append(val)
	e /= 300/10
	print("\t",e)
	Ecv.append(e)
	k.append(i)

for i in range(201,300,30):
	print(i)
	e = 0
	for j in range(0,300,10):
		# print(j)
		current = X.pop(0)
		val = Y.pop(0)
		# print(current,val)
		e += RBF(X,Y,current,val,i)
		# break
		# print(e)
		X.append(current)
		Y.append(val)
	e /= 300/10
	print("\t",e)
	Ecv.append(e)
	k.append(i)

for i in range(299,300):
	print(i)
	e = 0
	for j in range(0,300,10):
		# print(j)
		current = X.pop(0)
		val = Y.pop(0)
		# print(current,val)
		e += RBF(X,Y,current,val,i)
		# break
		# print(e)
		X.append(current)
		Y.append(val)
	e /= 300/10
	print("\t",e)
	Ecv.append(e)
	k.append(i)

plt.title("E_CV for various k for RBF Rule")
plt.xlabel("k")
plt.ylabel("E_CV")
plt.plot(k,Ecv)
plt.show()
 



