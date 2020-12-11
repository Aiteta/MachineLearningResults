import random
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math
import time

def tanh_prime(x):
	return 1-x**2

def identity(s):
	return s

def identity_prime(x):
	return 1

def forwardProp(x_og,W_all,d_all,L,out):
	X = list()
	x = x_og
	for i in range(L):
		x = list(x)
		x.insert(0,1)
		# print(x)
		# x = np.concatenate([[1],x])
		X.append(x)
		x = np.dot(np.transpose(W_all[i]),x)
		if (i != L-1):
			x = list(map(math.tanh,x))
	X.append(out(x[0]))
	return X

def forwardProp1(X_all,W_all,L,out):
	X = list()
	tanh_v = np.vectorize(math.tanh)
	x = np.array(X_all).T
	for i in range(L):
		x = np.insert(x,0,[1],axis = 0)
		X.append(x)
		x = np.dot(np.transpose(W_all[i]),x)
		if (i != L-1):
			x = tanh_v(x)
	X.append(out(x))
	return X

def backwardProp(X,W_all,y,out_prime,L):
	Delta = list()
	Delta.append(np.array([2*(X[-1]-y)*out_prime(X[-1])]))
	for i in range(L-1):
		l = len(X)-2-i
		x = X[l][1:]
		if (i == 0):
			print(x)
		delta = list()
		theta = list(map(tanh_prime,x))
		temp = np.dot(W_all[l][1:,:],Delta[-1])
		Delta.append(np.multiply(temp,theta))
	return Delta

def backwardProp1(X,W_all,Y_all,out_prime,L):
	Delta = list()
	Delta.append(2*(X[-1]-Y_all)*out_prime(X[-1]))
	for i in range(L-1):
		l = len(X)-2-i
		x = X[l][1:,:]
		theta = np.vectorize(tanh_prime)(x)
		temp = np.dot(W_all[l][1:,:],Delta[-1])
		Delta.append(np.multiply(temp,theta))
	return Delta

def E_in(X_all,Y_all,W_all,d_all,L,out,out_prime):
	global time_forward
	global time_backward
	global time_outer
	Ein = 0
	G_all = list(map(lambda w: np.zeros(w.shape),W_all))
	N = len(X_all)
	start = time.time()
	X = forwardProp1(X_all,W_all,L,out)
	time_forward += time.time()-start
	start = time.time()
	Delta = backwardProp1(X,W_all,Y_all,out_prime,L)
	time_backward += time.time()-start
	error_v = np.vectorize(lambda x: x**2/N)
	Ein = sum(error_v(X[-1]-Y_all)[0])
	start = time.time()
	for l in range(L):
		for i in range(len(X_all)):
			G_all[l] += np.outer(X[l][:,i],Delta[L-1-l][:,i])/N
	time_outer += time.time()-start
	return (Ein,G_all)

def E_in_noGrad(X_all,Y_all,W_all,d_all,L,out,out_prime):
	global time_forward
	Ein = 0
	N = len(X_all)
	start = time.time()
	X = forwardProp1(X_all,W_all,L,out)
	time_forward += time.time()-start
	error_v = np.vectorize(lambda x: x**2/N)
	Ein = sum(error_v(X[-1]-Y_all)[0])
	# Ein = 0
	# N = len(X_all)
	# for n in range(len(X_all)):
	# 	x = X_all[n]
	# 	y = Y_all[n]
	# 	X = forwardProp(x,W_all,d,L,out)
	# 	Ein += ((X[-1]-y)**2)/N
	return Ein

def VarGradDes(X_all,Y_all,W_all,d_all,L,numIt):
	alpha = 1.075
	beta = 0.65
	eta = 0.5
	Ein = 0
	new = 0
	Ein_all = list()
	for i in range(numIt):
		# print(Ein)
		# print(new)
		global time_total
		global time_other
		if (i %100 == 0):
			print(i)
		start = time.time()
		(Ein, G_all) = E_in(X_all,Y_all,W_all,d_all,L,np.vectorize(identity),np.vectorize(identity_prime))
		time_total += time.time()-start
		Ein_all.append(Ein)
		temp = list()
		for i in range(len(W_all)):
			temp.append(W_all[i]-eta*G_all[i]) 
		# temp = list(map(lambda w,g:w-eta*g,W_all,G_all))
		start = time.time()
		new = E_in_noGrad(X_all,Y_all,temp,d_all,L,np.vectorize(identity),np.vectorize(identity_prime))
		time_other += time.time()-start

		if (Ein >= new):
			W_all = temp
			eta = alpha*eta
		else:
			eta = beta*eta
		if (i == 1000):
			break
	return (W_all,Ein,Ein_all)

def NueralNetwork(X,Y,L,d_all,numIt):
	W_all = list()
	for i in range(L):
		W = list()
		for j in range(d[i+1]):
			w = list()
			for k in range(d[i]+1):
				w.append(0.2*(random.random()-0.5))
			W.append(w)
		W = np.array(W)
		W_all.append(np.transpose(W))

	(W_all,Ein,Ein_all) = VarGradDes(X,Y,W_all,d,L,numIt)
	return (W_all,Ein,Ein_all)

time_forward = 0
time_backward = 0
time_outer = 0
time_total = 0
time_other = 0


trainFile = open("train.txt")
trainLines = trainFile.readlines()

trainingData = list()
X = list()
Y = list()
for i in trainLines:
	data = i.split()
	X.append(np.array([float(data[1]),float(data[2])]))
	if (int(data[0]) == 1):
		Y.append(1)
		plt.plot(float(data[1]),float(data[2]),'ob')
	else:
		Y.append(-1)
		plt.plot(float(data[1]),float(data[2]),'xr')

m = 10
L = 2
d = [2,m,1]
numIt = 1000

(W_all,Ein,Ein_all) = NueralNetwork(X,Y,L,d,numIt)
print(Ein)
iteration = list()
for i in range(numIt):
	iteration.append(i+1)

print(time_forward)
print(time_backward)
print(time_outer)
print(time_total)
print(time_other)

# f = open("outAlex.txt",'w')
# for i in range(len(iteration)):
# 	f.write("{}\n".format(Ein_all[i]))

# for i in range(51):
# 	for j in range(51):
# 		x = [i/25-1,j/25-1]
# 		(x,X,S) = forwardProp(x,W_all,d,L,np.sign)
# 		y = X[-1]
# 		if (y < 0):
# 			plt.plot(i/25-1,j/25-1,'sr', alpha = 0.2)
# 		else:
# 			plt.plot(i/25-1,j/25-1,'sb', alpha = 0.2)
# plt.show()


# plt.loglog(iteration,Ein_all)






