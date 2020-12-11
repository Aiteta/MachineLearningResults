from math import sin,pi,cos
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random

def f(x,y):
	return x**2+2*(y**2)+2*sin(2*pi*x)*sin(2*pi*y)
def del_f(x,y):
	return [2*x+4*pi*cos(2*pi*x)*sin(2*pi*y),4*y+4*pi*sin(2*pi*x)*cos(2*pi*y)]

x   = 0.1
y   = 0.1
eta = 0.1
itr = list()
out = list()
itr.append(0)
out.append(f(x,y))

for i in range(50):
	g = del_f(x,y)
	v = [-g[0],-g[1]] 
	x += eta*v[0]
	y += eta*v[1]
	itr.append(i+1)
	out.append(f(x,y))

print(f(x,y))
plt.plot(itr,out)
plt.xlabel("Iterations")
plt.ylabel("f(x,y)")
plt.title("Gradient Descent Eta = 0.1")

plt.show()

