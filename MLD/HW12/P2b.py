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

# Weight Decay with use of parametrization parameter lambda
# Return final Weights, final in-sample error 
# and the in-sample error over iterations
# In addition, Augmented Error
def WeightDecay(X,Y,W_all,L,numIt,lamb):
	# Initial Setup
	alpha = 1.08
	beta = 0.7
	eta = 0.5
	Ein = 0
	Eaug = 0
	new = 0
	Ein_all = list()
	Eaug_all = list()
	identity_v = np.vectorize(identity)
	identity_prime_v = np.vectorize(identity_prime)
	N = len(X)
	# Iterate over number of specified iterations
	for i in range(numIt):
		# Calculate In-Sample error and gradient
		(Ein, G_all) = E_in(X,Y,W_all,L,identity_v,identity_prime_v)
		Ein_all.append(Ein)
		# Calculate Augmented Error
		Eaug = Ein
		for i in range(len(W_all)):
			Eaug += lamb/N*np.sum(np.vectorize(lambda x: x**2)(W_all[i]))
		# Calculate new weights and check 
		# if the new weights perform better or worse
		# Based on Augmented Error
		Eaug_all.append(Eaug)
		temp = list(map(lambda w,g: w-eta*(g+2*lamb/N*w),W_all,G_all))
		new = E_in_noGrad(X,Y,temp,L,identity_v,identity_prime_v)
		newaug = new
		for i in range(len(W_all)):
			newaug += lamb/N*np.sum(np.vectorize(lambda x: x**2)(temp[i]))
		if (Eaug > newaug):
			W_all = temp
			eta = alpha*eta
		else:
			eta = beta*eta
	return (W_all,Ein,Ein_all,Eaug,Eaug_all)

# Nueral network, perform learning on given dataset over predefined
# Iterations and user-defined hidden layers
# Return final weights and in-sample error analysis
def NueralNetwork(X,Y,L,numIt,lamb):
	# Initialize weights
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
	(W_all,Ein,Ein_all,Eaug,Eaug_all) = WeightDecay(X,Y,W_all,L,numIt,lamb)
	return (W_all,Ein,Ein_all,Eaug,Eaug_all)

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
lamb = 0.01/len(X)

(W_all,Ein,Ein_all,Eaug,Eaug_all) = NueralNetwork(X,Y,L,numIt,lamb)
print(Ein)
print(Eaug)

# Output data into text files to not lose precisious data
f = open("outError2bEaug.txt",'w')
h = open("outError2bEin.txt",'w')
for i in range(numIt):
	f.write("{}\n".format(Eaug_all[i]))
	h.write("{}\n".format(Ein_all[i]))
f.close()
h.close()
f = open("outWeights2b.txt",'w')
out = ""
for i in range(len(W_all)):
	for j in range(len(W_all[i])):
		for k in range(len(W_all[i][j])):
			out += str(W_all[i][j][k]) + " "
		out += '\n'
	out += '\n'
f.write(out)
f.close()