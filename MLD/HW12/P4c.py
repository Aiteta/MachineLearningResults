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
	else:
		Y.append(-1)

# Organize Data for QP solve except QD and p
N = len(Y)
G = np.zeros((N,N)) # -Ad(aplha) <= c wasn't working
h = np.zeros(N)     # So used Ad(alpha) = c, specifying a lower
A = np.array(Y)     # AD, and upper bound for alpha
b = np.zeros(1)     # c
lower = np.zeros(N) # Lower Bound
upper = np.ones(N)*C# Upper Bound
lower = np.zeros(N-1)
upper = np.ones(N-1)*C
Ecv_all = list()
C_all = list()
# Perform Ecv over 10^-2 <= C < 30
# Perform 10-fold Ecv
# Save Ecv for each removed data point
for i in range(31):
	C = 10**(i/30*3.5-2)
	C_all.append(C)
	Ecv = 0
	for j in range(len(X)//10):
		x = X.pop(0)
		y = Y.pop(0)
		QD = list()
		for i in range(len(X)):
			w = list()
			for j in range(len(X)):
				# print(np.dot(X[i],X[j]))
				if (i != j):
					w.append(Y[i]*Y[j]*K(X[i],X[j]))
				else:
					w.append(Y[i]*Y[j]*K(X[i],X[j])+10**(-8))
			QD.append(w)
		# Create QD and p and do QP solve
		P = np.array(QD) # QD
		A = np.array(Y)  # p
		alpha = solve_qp(P,q,G,h,A,b,lower,upper)
		# Find Support Vectors
		sup_vec_index = list()
		for i in range(len(alpha)):
			if (alpha[i] > tol):
				sup_vec_index.append(i)
		# Calculate Bias
		d = Y[sup_vec_index[0]]
		for i in range(len(sup_vec_index)):
			j = sup_vec_index[i]
			d -= Y[j]*alpha[j]*K(X[j],X[sup_vec_index[0]])
		# Classify removed 
		guess = 0+d
		for k in range(len(sup_vec_index)):
			l = sup_vec_index[k]
			guess += Y[l]*alpha[l]*K(X[l],np.array([i/25-1,j/25-1]))
		if (np.sign(guess) != y):
			Ecv += 2
		X.append(x)
		Y.append(y)
	Ecv_all.append(Ecv/N*10)

# Plot Ecv vs parametrization value C
plt.semilogx(C_all,Ecv_all)
plt.title("SVM: Ecv vs. C (Regularization)")
plt.ylabel("Error")
plt.xlabel("C (Regularization)")

plt.show()



