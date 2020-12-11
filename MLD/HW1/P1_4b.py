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

seed = int(input("Seed: "))
num  = int(input("Number: "))


random.seed(seed)

func = np.array([25,-12])
target = -690
x_all = list() 
y_all = list()

# n = np.arange(27.6,75.6)
n = np.arange(0,100)

plt.plot(n,-1*func[0]/func[1]*n-target/func[1], '-k', label = 'Target Function: f')

for i in range(num):
	x1 = random.random()*100
	x2 = random.random()*100
	x_all.append(np.array([x1,x2]))
	y_all.append(np.sign(x1*func[0]+x2*func[1]+target))
	if y_all[len(y_all)-1] > 0:
		plt.plot(x1,x2,'xr')
	else:
		plt.plot(x1,x2,'ob')

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

plt.plot(n,-1*g[0]/g[1]*n-cur_tar/g[1], '-g', label = 'Final Hypothesis: g')
plt.legend()
plt.ylim((0,100))
plt.xlim((0,100))

# plt.title("Problem 1.4(a)")
# plt.xlabel("x1")
# plt.ylabel("x2")
print(total)
plt.show()