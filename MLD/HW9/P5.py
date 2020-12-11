import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math
import random

def Legendre(x,num):
	if (num < 0):
		return 1
	if (num == 0):
		return 1
	elif (num == 1):
		return x
	prev = 1
	curr = x
	k = 2
	while (k <= num):
		temp = curr
		curr = (2*k-1)/k*x*curr-(k-1)/k*prev
		prev = temp
		k += 1
	return curr

def createZ(x,order):
	Z = list()
	for i in range(order+1):
		for j in range(i+1):
			Z.append(Legendre(x[1],i-j)*Legendre(x[2],j))
	# print(Z)
	return Z

dataFile = open("train.txt")
digitLines = dataFile.readlines()

train = list()
for i in digitLines:
	line = i.split()
	train.append((int(line[0]),float(line[1]),float(line[2])))

X = list()
y = list()
j = 0
for i in train:
	if (i[0] == 1):
		y.append(1)
	else:
		y.append(-1)
	X.append(createZ(i,8))

Z = np.array(X)
y = np.array(y)
print(Z.shape)

ZTZ = np.matmul(np.transpose(Z),Z)
ZTZ_1ZT = np.matmul(np.linalg.inv(ZTZ + 2.17*np.identity(45)),np.transpose(Z))
w_reg = np.matmul(ZTZ_1ZT,y)

for i in range(70):
	a = 2*i/69-1
	for j in range(70):
		b = 2*j/69-1
		g = np.matmul(createZ([1,a,b],8),w_reg)
		if (g < 0):
			plt.plot(a,b,'sr', alpha = 0.1)
		else:
			plt.plot(a,b,'sb', alpha = 0.1)

testFile = open("converted.txt")
testLines = testFile.readlines()
for i in testLines:
	line = i.split()
	if (int(line[0]) == 1):
		plt.plot(float(line[1]),float(line[2]),'ob')
	else:
		plt.plot(float(line[1]),float(line[2]),'xr')

plt.title("Feature Plot Regression Lambda=2.17 and Test Data")
plt.xlabel('Average Intensisty')
plt.ylabel('Vertical-Horizontal Ratio')

plt.show()

