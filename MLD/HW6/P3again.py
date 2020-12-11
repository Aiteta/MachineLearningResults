import random
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math

def add(w,x,y):
	for i in range(len(x)):
		w[i] += x[i]*y

def mult(x,w):
	out = 0
	for i in range(len(x)):
		out += x[i]*w[i]
	return out

def check(X,Y,w):
	start = int(random.random()*len(X)//1)
	# print(start)
	for i in range(start,len(X)):
		if (sign(mult(X[i],w)) != sign(Y[i])):
			return i
	for i in range(start):
		if (sign(mult(X[i],w)) != sign(Y[i])):
			return i
	return -1

def sign(x):
	if (x < 0):
		return -1
	elif (x > 0):
		return 1
	return 0

sepX = list()
numIterY = list()

for sep in range(25):
	sep = (sep+1)*0.2
	print(sep)
	rad = 10
	thk = 5

	final = 0
	for i in range(10000):
		if (i % 1000 == 0):
			print(i)
		X = list()
		Y = list()
		for i in range(2000):
			random1 = 2*math.pi*random.random()
			random2 = random.random()
			if (random1 < math.pi):
				x = math.cos(random1)*(rad+thk*(random2))
				y = math.sin(random1)*(rad+thk*(random2))
				X.append([1,x,y])
				Y.append(1)
				# plt.plot(x,y,'ob')
			else:
				x = thk/2+rad + math.cos(random1)*(rad+thk*(random2))
				y = -sep + math.sin(random1)*(rad+thk*(random2))
				X.append([1,x,y])
				Y.append(-1)
				# plt.plot(x,y,'xr')

	# plt.show()
		if (i%1000 == 0):
			print(i)
		tryer = 0
		w = [0,0,0]
		while (True):
			index = check(X,Y,w)
			if (index == -1):
				break;
			add(w,X[index],Y[index])
			tryer += 1
		final += tryer
	final /= 10000
	sepX.append(sep)
	numIterY.append(final)


plt.plot(sepX,numIterY)
plt.xlabel('Seperation')
plt.ylabel('Average Number of Iterations')
plt.title('Problem 3.2 Average Number of Iterations vs. Seperation')
plt.show()

