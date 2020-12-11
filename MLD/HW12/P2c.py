import random
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math
import time

# Derivative of tanh function
def tanh_prime(x):
	return 1-x**2

# Identity function
def identity(s):
	return s

# Derivative of Identity Function
def identity_prime(x):
	return 1

# Performs forward propogation
# Output of each Layer is recorded and output
def forwardProp(X_all,W_all,L,out):
	X = list()
	tanh_v = np.vectorize(math.tanh)
	x = np.array(X_all).T
	# Iterate through all Layers of NN,
	# All operations are performed on the entire data set at a times
	for i in range(L):
		x = np.insert(x,0,[1],axis = 0)
		X.append(x)
		x = np.dot(np.transpose(W_all[i]),x)
		if (i != L-1):
			x = tanh_v(x)
	X.append(out(x))
	return X

# Performed backward propogation
# Delta of each hidden layer and output layer is returned
# Which is used in calculating the gradient
def backwardProp(X,W_all,Y_all,out_prime,L):
	Delta = list()
	# Delta of output
	Delta.append(2*(X[-1]-Y_all)*out_prime(X[-1]))
	# Iteration through all hidden layers
	for i in range(L-1):
		l = len(X)-2-i
		x = X[l][1:,:]
		theta = np.vectorize(tanh_prime)(x)
		temp = np.dot(W_all[l][1:,:],Delta[-1])
		Delta.append(np.multiply(temp,theta))
	return Delta

# In-Sample Error is calculated
# Gradient of each weight matrix is returned
def E_in(X_all,Y_all,W_all,L,out,out_prime):
	Ein = 0
	G_all = list(map(lambda w: np.zeros(w.shape),W_all))
	N = len(X_all)
	# Forward Propogation
	X = forwardProp(X_all,W_all,L,out)
	# Backward Propogation
	Delta = backwardProp(X,W_all,Y_all,out_prime,L)
	error_v = np.vectorize(lambda x: x**2/N)
	Ein = sum(error_v(X[-1]-Y_all)[0])
	# Calculation of gradient
	for l in range(L):
		Xl = X[l].T
		Dl = Delta[L-1-l].T
		G_all[l] = sum(list(map(lambda x,d: np.outer(x,d)/N,Xl,Dl)))
	return (Ein,G_all)

# In-Sample Error is calculated
def E_in_noGrad(X_all,Y_all,W_all,L,out,out_prime):
	Ein = 0
	N = len(X_all)
	# Forward Propogation
	X = forwardProp(X_all,W_all,L,out)
	error_v = np.vectorize(lambda x: x**2/N)
	Ein = sum(error_v(X[-1]-Y_all)[0])
	return Ein

# Early Stopping
# Return final Weights, final in-sample error 
# and the in-sample error over iterations
# In addition, Test Error
# Final Weights return are based on best test error performance
def EarlyStopping(X,Y,W_all,L,numIt,size):
	# Initial Setup
	alpha = 1.075
	beta = 0.65
	eta = 0.5
	Ein = 0
	new = 0
	# Seperate Data into Validation Set and Training Set
	testX = list()
	testY = list()
	newX = list()
	newY = list()
	test_it = set(random.sample(range(len(X)),size))
	for i in range(len(X)):
		if (i in test_it):
			testX.append(X[i])
			testY.append(Y[i])
		else:
			newX.append(X[i])
			newY.append(Y[i])
	X = np.array(newX)
	Y = np.array(newY)
	testX = np.array(testX)
	testY = np.array(testY)
	# More initial setup
	Ein_all = list()
	Etest_all = list()
	bestEtest = -1
	bestW = W_all
	identity_v = np.vectorize(identity)
	identity_prime_v = np.vectorize(identity_prime)
	# Iterate over number of specified iterations
	for i in range(numIt):
		# print(Ein)
		# print(bestEtest)
		if (i %100 == 0):
			print(i)
		# 
		(Ein, G_all) = E_in(X,Y,W_all,L,identity_v,identity_prime_v)
		Etest = E_in_noGrad(testX,testY,W_all,L,identity_v,identity_prime_v)
		Ein_all.append(Ein)
		Etest_all.append(Etest)
		# Save weights if they perform the best on test error
		if (bestEtest == -1 or Etest < bestEtest):
			bestEtest = Etest
			bestW = W_all
		# Calculate new weights and check 
		# if the new weights perform better or worse
		temp = list(map(lambda w,g: w-eta*g,W_all,G_all))
		new = E_in_noGrad(X,Y,temp,L,identity_v,identity_prime_v)
		if (Ein > new):
			W_all = temp
			eta = alpha*eta
		else:
			eta = beta*eta
	return (bestW,Ein,Ein_all,bestEtest,Etest_all)

# Nueral network, perform learning on given dataset over predefined
# Iterations and user-defined hidden layers
# Return final weights and in-sample error analysis
def NueralNetwork(X,Y,L,numIt,size):
	# Initialize Weights
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
	# LEARN
	(bestW,Ein,Ein_all,bestEtest,Etest_all) = EarlyStopping(X,Y,W_all,L,numIt,size)
	return (bestW,Ein,Ein_all,bestEtest,Etest_all)

# Read in Digits training data
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

# Initialize Parameters
m = 10
L = 2
d = [2,m,1]
numIt = 2000000
validationSize = 50

(bestW,Ein,Ein_all,bestEtest,Etest_all) = NueralNetwork(X,Y,L,numIt,validationSize)
print(bestEtest)
print(Ein)

# Output data into text files to not lose precisious data
f = open("outError2cEtest.txt",'w')
h = open("outError2cEin.txt",'w')
for i in range(numIt):
	f.write("{}\n".format(Etest_all[i]))
	h.write("{}\n".format(Ein_all[i]))
f.close()
h.close()
f = open("outWeights2c.txt",'w')
out = ""
for i in range(len(bestW)):
	for j in range(len(bestW[i])):
		for k in range(len(bestW[i][j])):
			out += str(bestW[i][j][k]) + " "
		out += '\n'
	out += '\n'
f.write(out)
f.close()