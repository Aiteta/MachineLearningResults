import random
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math
import time
from qpsolvers import solve_qp

# Kernel Function
def K(x1,x2):
	return (1+np.dot(x1,x2))**8

# Initial setup
C = 18
tol = 10**(-8)
# Read in training data and plot
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
# Create QD
QD = list()
for i in range(len(X)):
	q = list()
	for j in range(len(X)):
		# print(np.dot(X[i],X[j]))
		if (i != j):
			q.append(Y[i]*Y[j]*K(X[i],X[j]))
		else:
			q.append(Y[i]*Y[j]*K(X[i],X[j])+10**(-8))
	QD.append(q)

# Organize Data for QP solve
N = len(Y)
P = np.array(QD)    # QD
q = np.ones(N)*-1   # p
G = np.zeros((N,N)) # -Ad(aplha) <= c wasn't working
h = np.zeros(N)     # So used Ad(alpha) = c, specifying a lower
A = np.array(Y)     # AD, and upper bound for alpha
b = np.zeros(1)     # c
lower = np.zeros(N) # Lower Bound
upper = np.ones(N)*C# Upper Bound
alpha = solve_qp(P,q,G,h,A,b,lower,upper)
# Find Support Vectors
sup_vec_index = list()
for i in range(len(alpha)):
	if (alpha[i] > tol):
		sup_vec_index.append(i)
# Calculate Bias
b = Y[sup_vec_index[0]]
for i in range(len(sup_vec_index)):
	j = sup_vec_index[i]
	b -= Y[j]*alpha[j]*K(X[j],X[sup_vec_index[0]])

# Iterate over -1<=x1<=1 and -1<=x2<=1 to plot decision boundary
for i in range(51):
	for j in range(51):
		y = 0+b
		for k in range(len(sup_vec_index)):
			l = sup_vec_index[k]
			y += Y[l]*alpha[l]*K(X[l],np.array([i/25-1,j/25-1]))
		y = np.sign(y)
		if (y < 0):
			plt.plot(i/25-1,j/25-1,'sr', alpha = 0.2)
		else:
			plt.plot(i/25-1,j/25-1,'sb', alpha = 0.2)
plt.show()

plt.title("SVM: C = 30")
plt.xlabel("Average Intensity")
plt.ylabel("Vertical-Horizontal Ratio")

# Read in test data and calculate out of sample error and outpu
testFile = open("test.txt")
testLines = testFile.readlines()

Etest = 0
for i in testLines:
	data = i.split()
	x = [float(data[1]),float(data[2])]

	ytrue = 0
	if (int(data[0]) == 1):
		ytrue = 1
	else:
		ytrue = -1
	y = 0+b
	for k in range(len(sup_vec_index)):
		l = sup_vec_index[k]
		y += Y[l]*alpha[l]*K(X[l],x)

	if (np.sign(y) != ytrue):
		Etest += 1
testFile.close()

Etest /= len(testLines)
print(Etest)