from math import sin,pi,cos
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random

def f(x,y):
	return x**2+2*(y**2)+2*sin(2*pi*x)*sin(2*pi*y)
def del_f(x,y):
	return [2*x+4*pi*cos(2*pi*x)*sin(2*pi*y),4*y+4*pi*sin(2*pi*x)*cos(2*pi*y)]
def conv(x,y,eta):
	for i in range(50):
		g = del_f(x,y)
		v = [-g[0],-g[1]] 
		x += eta*v[0]
		y += eta*v[1]
	return (x,y,f(x,y))


print(conv(0.1,0.1,0.01),conv(1,1,0.01),conv(-0.5,-0.5,0.01),conv(-1,-1,0.01))
print(conv(0.1,0.1,0.1),conv(1,1,0.1),conv(-0.5,-0.5,0.1),conv(-1,-1,0.1))
