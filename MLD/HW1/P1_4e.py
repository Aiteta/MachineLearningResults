import numpy as np
from matplotlib import pyplot as plt
import random

def check(x_all,y_all,g,cur_tar):
	c = int(random.random()*(len(x_all)-1))
	for i in range(c,len(x_all)):
		if (np.sign(x_all[i][0]*g[0]+x_all[i][1]*g[1]+cur_tar)) != y_all[i]:
			return i
	for i in range(c):
		if (np.sign(x_all[i][0]*g[0]+x_all[i][1]*g[1]+cur_tar)) != y_all[i]:
			return i
	return -1

def mL(seed,n):
	random.seed(seed)

	func = np.array([25,-12])
	target = -690
	x_all = list() 
	y_all = list()


	for i in range(n):
		x1 = random.random()*100
		x2 = random.random()*100
		x_all.append(np.array([x1,x2]))
		y_all.append(np.sign(x1*func[0]+x2*func[1]+target))

	g = np.array([0,0])
	cur_tar = 0
	i = check(x_all,y_all,g,cur_tar)
	total = 0
	while (i != -1):
		total += 1
		g = g + y_all[i]*x_all[i]
		cur_tar = cur_tar + y_all[i]
		# print(cur_tar,g,i)
		i = check(x_all,y_all,g,cur_tar)

	return total



seed = int(input("Seed: "))

n = 1
t = mL(seed,n)
x = []
y = []
while (t < 1000000):
	print(n)
	x.append(n)
	y.append(t)
	n *= 2
	t = mL(seed,n)
print(n)

plt.loglog(np.array(x),np.array(y))
plt.show()
