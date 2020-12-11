import numpy as np
from matplotlib import pyplot as plt
import random

def repeat(thelist):
	out = list()
	for i in range(len(thelist)):
		out.append(thelist[i])
		out.append(thelist[i])
	out.pop()
	return out

random.seed(69)

odds = 0.5
sampleSize = 1000
iterations = 100000
n_bins = 10

nu1 = list()
nurand = list()
numin = list()

for n in range(iterations):
	mini = -1
	val = 0
	coins = list()
	for i in range(sampleSize):
		total = 0
		for j in range(n_bins):
			if (random.random() > odds):
				total += 1
		coins.append(total)
		if (mini == -1 or val > total):
			mini = i
			val = total

	nu1.append(coins[0]/10)
	nurand.append(coins[int(random.random()*sampleSize)]/10)
	numin.append(val/10)

	if (n %1000 == 0):
		print(n)

# axs[0].hist(nu1, bins = len(set(nu1)))
# axs[1].hist(nurand, bins = len(set(nurand)))
# axs[2].hist(numin, bins = len(set(numin)))

# plt.show()

nu1eps = dict()
nurandeps = dict()
numineps = dict()

for i in range(11):
	i = i/10
	nu1eps[i] = 0
	nurandeps[i] = 0
	numineps[i] = 0

for i in range(iterations):
	nu1eps[nu1[i]] += 1
	nurandeps[nurand[i]] += 1
	numineps[numin[i]] += 1
# print(nu1eps)

for i in range(11):
	i = i/10
	nu1eps[i] /= iterations
	nurandeps[i] /= iterations
	numineps[i] /= iterations

epsx = [0,0.1,0.2,0.3,0.4,0.5]
epsy1 = list()
epsyrand = list()
epsymin = list()

prev1 = 0
prevrand = 0
prevmin = 0
for i in epsx:
	# print(prev1)
	if (i == 0):
		prev1 += nu1eps[round(odds+i,2)]
	else:
		prev1 += nu1eps[round(odds+i,2)]  + nu1eps[round(odds-i,2)]
	epsy1.append(1-prev1)
	if (i == 0):
		prevrand += nurandeps[round(odds+i,2)]
	else:
		prevrand += nurandeps[round(odds+i,2)] + nurandeps[round(odds-i,2)]
	epsyrand.append(1-prevrand)
	if (i == 0):
		prevmin += numineps[round(odds+i,2)]
	else:
		prevmin += numineps[round(odds+i,2)] + numineps[round(odds-i,2)]
	epsymin.append(1-prevmin)


# print(epsymin)
epsx = np.array([0.0,0.1,0.1,0.2,0.2,0.3,0.3,0.4,0.4,0.5,0.5])
epsy1 = repeat(epsy1)
epsy1 = np.array(epsy1)
epsyrand = repeat(epsyrand)
epsyrand = np.array(epsyrand)
epsymin = repeat(epsymin)
epsymin = np.array(epsymin)

epsx1 = np.arange(0,0.51,0.01)
epsytrue = []
for i in range(51):
	epsytrue.append(2*2.7182818**(-(epsx1[i]**2)*n_bins))
epsytrue = np.array(epsytrue)


plt.title("Epsilon Probability Distribution for nu of 3 coins\n(N=10), (M=1000), (Iter=100,000)")
plt.xlabel("Epsilon = |mu-nu|")
plt.ylabel("Probability")

plt.plot(epsx,epsy1, ':r', label = 'First Coin Epsilon Distribution')
plt.plot(epsx,epsyrand, '--b', label = 'Random Coin Epsilon Distribution')
plt.plot(epsx,epsymin, '-.p', label = 'Minimum nu Coin Epsilon Distribution')
plt.plot(epsx1,epsytrue, '-k', label = 'Hoeffding Epsilon Distribution')

plt.legend()

# axs[0].plot(epsx,epsy1)
# axs[0].plot(epsx,epsytrue)
# axs[1].plot(epsx,epsyrand)
# axs[1].plot(epsx,epsytrue)
# axs[2].plot(epsx,epsymin)
# axs[2].plot(epsx,epsytrue)

plt.show()



# nu1d = dict()
# nurandd = dict()
# numind = dict()

# for i in range(11):
# 	nu1d[i] = 0
# 	nurandd[i] = 0
# 	numind[i] = 0

# for i in range(iterations):
# 	nu1d[nu1[i]] += 1
# 	nurandd[nurand[i]] += 1
# 	numind[numin[i]] += 1

# nu1arr = np.array(sorted(nu1d.keys()))
# nurandarr = np.array(sorted(nurandd.keys()))
# numinarr = np.array(sorted(numind.keys()))

