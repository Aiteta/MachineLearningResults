import math
import numpy as np

def tanh_prime(x):
	return 1-x**2

def forwardProp(x,W_all,d_all,L,out):
	X = list()
	S = list()
	for i in range(L):
		X.append(list(x))
		new = [1]
		for j in range(len(x)):
			new.append(x[j])
		x = np.transpose(np.array(new))
		d = d_all[i]
		W = W_all[i]
		# print('d',d,'x',x,'w',W)
		s = np.dot(np.transpose(W),x)
		S.append(list(s))
		x = list()
		if (i != L-1):
			for j in range(len(s)):
				x.append(math.tanh(s[j]))
		else:
			x = list(s)
	X.append(out(x[0]))
	return (x,X,S)

def backwardProp(X,S,W_all,y,out_prime,L):
	Delta = list()
	Delta.append(np.array([2*(X[len(X)-1]-y)*out_prime(X[len(X)-1])]))
	for i in range(L-1):
		l = len(X)-2-i
		x = np.transpose(X[l])
		delta_1 = Delta[-1]
		W = W_all[l][1:,:]
		theta = list()
		for j in range(len(x)):
			theta.append(1-x[j]**2)
		delta = list()
		temp = np.dot(W,delta_1)
		for j in range(len(temp)):
			delta.append(temp[j]*theta[j])
		Delta.append(np.array(delta))
	return Delta



m = 2
L = 3
d = [1,2,1,1]
x = [2]
y = 1

W_all = list()
W1 = np.array([[0.1,0.2],[0.3,0.4]])
W2 = np.array([[0.2],[1],[-3]])
W3 = np.array([[1],[2]])
W_all = [W1,W2,W3]
# for i in range(L):
# 	W = list()
# 	for j in range(d[i+1]):
# 		w = list()
# 		for k in range(d[i]+1):
# 			w.append(0.25)
# 		W.append(w)
# 	W = np.array(W)
# 	W_all.append(np.transpose(W))

(s,X,S) = forwardProp(x,W_all,d,L,math.tanh)
h = math.tanh(s[0])
print(x,h)
print(X)
print(S)
print(W_all)

# print()
Delta = backwardProp(X,S,W_all,y,tanh_prime,L)
print(Delta)







