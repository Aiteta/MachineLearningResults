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
	return Z

dataFile = open("train.txt")
testFile = open("test.txt")
testLines = testFile.readlines()
digitLines = dataFile.readlines()

train = list()
for i in digitLines:
	line = i.split()
	train.append((int(line[0]),float(line[1]),float(line[2])))

testX = list()
testY = list()
for i in testLines:
	line = i.split()
	testX.append(createZ((int(line[0]),float(line[1]),float(line[2])),8))
	if (int(line[0])==1):
		testY.append(1)
	else:
		testY.append(-1)
Z1 = np.array(testX)
testY = np.array(testY)

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

print(Z1.shape)
print(Z.shape)

lamb_x = list()
E_CV_y = list()
E_in_y = list()
E_out_y = list()
min_lam = -1
minfuck = 0
ZTZ = np.matmul(np.transpose(Z),Z)
for i in range(401):
	lamb = i*0.01
	print(lamb)
	lamb_x.append(lamb)
	ZTZ_1ZT = np.matmul(np.linalg.inv(ZTZ + lamb*np.identity(45)),np.transpose(Z))
	H = np.matmul(Z,ZTZ_1ZT)
	w_reg = np.matmul(ZTZ_1ZT,y)
	y_hat_1 = np.matmul(Z1,w_reg)
	y_hat = np.matmul(H,y)
	E_CV = 0
	E_in = 0
	E_out = 0
	for j in range(300):
		E_CV+=((np.sign(y_hat[j])-y[j])/(1-H[j,j]))**2
		E_in += (np.sign(y_hat[j])-y[j])**2
		# print(np.sign(y_hat[j])-y[j])
	k = 0
	for j in y_hat_1:
		E_out += (np.sign(j)-testY[k])**2
		k += 1
	E_CV_y.append(E_CV/300)
	E_in_y.append(E_in/300)
	E_out_y.append(E_out/k)
	
print(E_out_y)
print(min_lam)
plt.plot(lamb_x,E_CV_y,label="E_CV")
# plt.semilogy(lamb_x,E_in_y,label="E_in")
plt.plot(lamb_x,E_out_y,label="E_test(w_reg(lambda))")
plt.legend()
plt.xlabel("lambda")
plt.ylabel("Error")
plt.title("log E_CV and log E_test(w_reg(lambda)) vs lambda")


plt.show()

