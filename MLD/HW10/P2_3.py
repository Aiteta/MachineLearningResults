import random
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math

def distance(x1,x2):
	return ((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)**0.5

def returnFurthest(closest,k,X,x):
	closer = list()
	for i in range(len(closest)):
		if (distance(X[k],x) < distance(X[closest[i]],x)):
			closer.append(i)
	if (len(closer) == 0):
		return -1
	furthest = closer[0]
	dist = distance(X[closest[furthest]],x)
	for i in range(1,len(closer)):
		if (dist < distance(X[closest[closer[i]]],x)):
			furthest = closer[i]
			dist = distance(X[closest[closer[i]]],x)
	# print(furthest)
	return furthest

rad = 10
thk = 5
sep = 5

X = list()
Y = list()

for i in range(2000):
	random1 = 2*math.pi*random.random()
	random2 = random.random()
	if (random1 < math.pi):
		x = math.cos(random1)*(rad+thk*(random2))
		y = math.sin(random1)*(rad+thk*(random2))
		X.append([x,y])
		Y.append(-1)
	else:
		x = thk/2+rad + math.cos(random1)*(rad+thk*(random2))
		y = -sep + math.sin(random1)*(rad+thk*(random2))
		X.append([x,y])
		Y.append(1)

y = Y
print(len(X))

for i in range(100):
	print(i)
	for j in range(100):
		x = [-15+42.5*i/100,-20+35*j/100]
		closest = [0,1,2]
		for k in range(1,len(X)):
			furthest = returnFurthest(closest,k,X,x)
			if (furthest != -1):
				closest[furthest] = k
		if (y[closest[0]] + y[closest[1]] + y[closest[2]] < 0):
			plt.plot(x[0],x[1],'sr', alpha = 0.1)
		else:
			plt.plot(x[0],x[1],'sb', alpha = 0.1)


for i in range(len(X)):
	# print(X[i])
	if (y[i] == -1):
		plt.plot(X[i][0],X[i][1],'xr')
	else:
		plt.plot(X[i][0],X[i][1],'ob')

plt.title('3-NN')
plt.xlabel('x_1')
plt.ylabel('x_2')

plt.show()