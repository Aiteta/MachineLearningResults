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


# Input Parameters
m = 2
L = 2
d = [2,m,1]
x = [[1,2]]
y = [[1]]

# Initial Weights constructed
W_all = list()
for i in range(L):
	W = list()
	for j in range(d[i+1]):
		w = list()
		for k in range(d[i]+1):
			w.append(0.25)
		W.append(w)
	W = np.array(W)
	W_all.append(np.transpose(W))

# Vectorize Output layer functions
identity_v = np.vectorize(identity)
identity_prime_v = np.vectorize(identity_prime)
tanh_v = np.vectorize(math.tanh)
tanh_prime_v = np.vectorize(tanh_prime)

(Ein, G_all) = E_in(x,y,W_all,L,identity_v,identity_prime_v)
print(Ein,G_all)
(Ein, G_all) = E_in(x,y,W_all,L,tanh_v,tanh_prime_v)
print(Ein,G_all)


