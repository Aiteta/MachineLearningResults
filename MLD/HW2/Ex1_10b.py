import numpy as np
from matplotlib import pyplot as plt
import random

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
		for j in range(10):
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

fig, axs = plt.subplots(1, 3, sharey=True, sharex = True, tight_layout=True)

axs[0].hist(nu1, bins = len(set(nu1)))
axs[1].hist(nurand, bins = len(set(nurand)))
axs[2].hist(numin, bins = len(set(numin)))
axs[0].set_xlabel("nu")
axs[1].set_xlabel("nu")
axs[2].set_xlabel("nu")

axs[0].set_ylabel("Number of Coins")
axs[1].set_ylabel("Number of Coins")
axs[2].set_ylabel("Number of Coins")

axs[0].set_title("C_1")
axs[1].set_title("C_random")
axs[2].set_title("C_minimum")

fig.suptitle("nu Distribution for different coins \n(N=10), (M=1000), (Iter=100,000)")



plt.show()

