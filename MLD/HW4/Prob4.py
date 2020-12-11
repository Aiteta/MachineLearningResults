import numpy as np
from matplotlib import pyplot as plt
import random

i_tot = 40*40;
j_tot = 40;

a_tot = 0.0
b_tot = 0.0

x = np.linspace(-1,1,3)

for i in range(i_tot+1):
	x1 = random.random()*2-1
	x2 = random.random()*2-1
	if (x2 == x1):
		continue
	y1 = x1**2
	y2 = x2**2
	a = (y2-y1)/(x2-x1)
	b = (y1-a*x1)
	a_tot += a
	b_tot += b
	plt.plot(x,a*x+b,'#08080808')

a = a_tot/((i_tot+1)*(j_tot+1))
b = b_tot/((i_tot+1)*(j_tot+1))

x = np.linspace(-1,1,50)
y = x.copy()
for i in range(50):
	y[i] = y[i]**2
plt.plot(x,y,'r',label = 'Target Function')
plt.plot(x,a*x+b,'g', label = 'g_bar')
plt.legend()
plt.title("Problem 2.24c")
plt.xlabel("x")
plt.ylabel("y")

print("g_avg = {}x+{}".format(a,b))
plt.show()

# bias = 0
# for i in range(1000):
# 	x1 = random.random()*2-1
# 	x2 = random.random()*2-1
# 	if (x2 == x1):
# 		continue
# 	y1 = x1**2
# 	y2 = x2**2
# 	a = (x2+x1)
# 	b = -x1*x2
# 	biasi = 0
# 	for j in range(10000):
# 		xi = random.random()*2-1
# 		biasi += (a*xi+b-(xi)**2)**2
# 	bias += biasi/10000

# print(bias/1000)


# x = np.linspace(-1,1,100)
# bias = 0
# for xi in x:
# 	bias += (-xi**2)**2
# bias /= 101
# bias *= 0.5
# print("bias = {}".format(bias))

# print("Hello\n")
# var = 0
# for i in range(1000):
# 	x1 = random.random()*2-1
# 	x2 = random.random()*2-1
# 	y1 = x1**2
# 	y2 = x2**2
# 	a = (y2-y1)/(x2-x1)
# 	b = (y1-a*x1)
# 	vari = 0
# 	for j in range(10000):
# 		xi = random.random()*2-1
# 		vari += ((a*xi+b)**2)
# 	var += vari/1000
# var /= 10000
# print("var = {}".format(var))

