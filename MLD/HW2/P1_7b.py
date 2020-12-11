import numpy as np
from matplotlib import pyplot as plt
import random

def factorial(n):
	out = 1
	while (n > 0):
		out *= n
		n -= 1
	return out

def choose(n,r):
	out = factorial(n)/factorial(n-r)/factorial(r)
	return out

def binom(n,r,prob):
	return choose(n,r)*(prob)**(r)*(1-prob)**(n-r)

def binomSum(bound,n,prob,start = 0):
	out = 0
	for i in range(start,bound+1):
		out += binom(n,i,prob)
	print(out)
	return out

P0_50 = 2*binomSum(0,6,0.5)
print(P0_50)
P0_33 = 2*binomSum(1,6,0.5)
print(P0_33)
P0_16 = 2*binomSum(2,6,0.5)
print(P0_16)
P0_00 = 2*binomSum(2,6,0.5) + binom(6,3,0.5)
print(P0_00)

P_5 = 2*P0_50 - P0_50*P0_50
print(P_5)
P_3 = 2*P0_33 - P0_33*P0_33
print(P_3)
P_1 = 2*P0_16 - P0_16*P0_16
print(P_1)
P_0 = 2*P0_00 - P0_00*P0_00
print(P_0)

x = np.array([0,0,1/6,1/6,1/3,1/3,1/2,1/2])
y = np.array([P_0,P_1,P_1,P_3,P_3,P_5,P_5,0])

x1 = np.arange(0,0.51,0.01)
y1 = []
for i in range(51):
	y1.append(2*2.7182818**(-(x1[i]**2)*6))
y1 = np.array(y1)

plt.plot(x,y, label = 'max(|mu-nu| > epsilon)')
plt.plot(x1,y1, label = 'Hoeffding Bound')
plt.title("Problem 1.7b")
plt.xlabel("Epsilon")
plt.ylabel("Probability")
plt.legend()

plt.show()
