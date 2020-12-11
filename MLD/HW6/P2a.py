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
	i = 0
	for i in range(len(X)):
		if (sign(mult(X[i],w)) != sign(Y[i])):
			return i
		i += 1
	return -1

def sign(x):
	if (x < 0):
		return -1
	elif (x > 0):
		return 1
	return 0

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
		X.append([1,x,y])
		Y.append(-1)
		plt.plot(x,y,'rx')
	else:
		x = thk/2+rad + math.cos(random1)*(rad+thk*(random2))
		y = -sep + math.sin(random1)*(rad+thk*(random2))
		X.append([1,x,y])
		Y.append(1)
		plt.plot(x,y,'ob')

w = [0,0,0]


while (True):
	index = check(X,Y,w)
	if (index == -1):
		break;
	add(w,X[index],Y[index])


xw = np.linspace(-15,27.5, 2)
yw = -w[1]/w[2]*xw-w[0]/w[2]
print(w)
plt.plot(xw,yw,'g',label='Final Hypothesis')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Problem 3.1a PLA')
plt.show()

