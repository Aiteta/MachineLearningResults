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




foundBrute = list()
foundOpt = list()

centers = [random.randrange(0,10000)]

for i in range(9):
	centers.append(findFurthest(centers,X))

regions = list()
for i in range(10):
	regions.append(list())
middles = list()

for i in range(10000):
	regions[closestCenter(centers,X[i],X)].append(i)

for i in range(len(regions)):
	mu = [0,0]
	for j in range(len(regions[i])):
		mu[0] += X[regions[i][j]][0]
		mu[1] += X[regions[i][j]][1]
	mu[0] /= len(regions[i])
	mu[1] /= len(regions[i])
	middles.append(mu)

radii = list()

# for i in regions[0]:
# 	plt.plot(X[i][0],X[i][1],'.b')

# for i in regions[1]:
# 	plt.plot(X[i][0],X[i][1],'.g')

# for i in regions[2]:
# 	plt.plot(X[i][0],X[i][1],'.r')

# for i in regions[3]:
# 	plt.plot(X[i][0],X[i][1],'.c')

# for i in regions[4]:
# 	plt.plot(X[i][0],X[i][1],'.m')

# for i in regions[5]:
# 	plt.plot(X[i][0],X[i][1],'.y')

# for i in regions[6]:
# 	plt.plot(X[i][0],X[i][1],'.k')

# for i in regions[7]:
# 	plt.plot(X[i][0],X[i][1],'.b')

# for i in regions[8]:
# 	plt.plot(X[i][0],X[i][1],'.g')

# for i in regions[9]:
# 	plt.plot(X[i][0],X[i][1],'.r')

# plt.show()

for i in range(len(regions)):
	rad = 0
	for j in range(len(regions[i])):
		dist = distance(middles[i],X[regions[i][j]])
		if (dist > rad):
			rad = dist
	radii.append(rad)

for i in range(len(middles)):
	print(middles[i],radii[i])

start = time.time()
for i in range(10000):
	if (not i%1000):
		print(i)
	foundOpt.append(findOpt(X,queryPoints[i],regions,middles,radii))
end = time.time()
print(end-start)

start = time.time()
for i in range(10000):
	if (not i%1000):
		print(i)
	foundBrute.append(findBrute(X,queryPoints[i]))
end = time.time()
print(end-start)

for i in range(len(foundBrute)):
	if (foundBrute[i] != foundOpt[i]):
		print(i,"ERROR", distance(X[foundBrute[i]],queryPoints[i]), distance(X[foundOpt[i]],queryPoints[i]))
		break











	