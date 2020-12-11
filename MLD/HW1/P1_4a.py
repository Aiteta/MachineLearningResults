import numpy as np
from matplotlib import pyplot as plt
import random

random.seed(70)

func = np.array([25,-12])
target = -690
x_all = list() 
y_all = list()

n = np.arange(27.6,75.6)
plt.plot(n,-1*func[0]/func[1]*n-target/func[1], '-k', label = 'Target Function: f')
plt.legend()


for i in range(20):
	x1 = random.random()*100
	x2 = random.random()*100
	x_all.append(np.array([x1,x2]))
	y_all.append(np.sign(x1*func[0]+x2*func[1]+target))
	if y_all[len(y_all)-1] > 0:
		plt.plot(x1,x2,'xr')
	else:
		plt.plot(x1,x2,'ob')

plt.title("Problem 1.4(a)")
plt.xlabel("x1")
plt.ylabel("x2")


plt.show()