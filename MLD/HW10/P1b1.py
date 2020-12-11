import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math

def convert(x):
	if (x[0] == 0 and x[1] < 0):
		return [(x[0]**2+x[1]**2)**(0.5),-math.pi/2]
	elif (x[0] == 0 and x[1] >= 0):
		return [(x[0]**2+x[1]**2)**(0.5),math.pi/2]
	else:
		return [(x[0]**2+x[1]**2)**(0.5),math.atan(x[1]/x[0])]

def distance(x1,x2):
	z1 = convert(x1)
	z2 = convert(x2)
	return ((z1[0]-z2[0])**2+(z1[1]-z2[1])**2)**0.5

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


X = [[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]]
y = [-1,-1,-1,-1,1,1,1]

for i in range(100):
	print(i)
	for j in range(100):
		x = [-3+6*i/100,-3+6*j/100]
		closest = [0]
		for k in range(1,7):
			furthest = returnFurthest(closest,k,X,x)
			if (furthest != -1):
				closest[furthest] = k
		if (y[closest[0]] < 0):
			plt.plot(x[0],x[1],'sr', alpha = 0.1)
		else:
			plt.plot(x[0],x[1],'sb', alpha = 0.1)

for i in range(len(X)):
	if (y[i] == -1):
		plt.plot(X[i][0],X[i][1],'xr')
	else:
		plt.plot(X[i][0],X[i][1],'ob')

plt.title('Non-Linear Transform 1-NN')
plt.xlabel('x_1')
plt.ylabel('x_2')

plt.show()

