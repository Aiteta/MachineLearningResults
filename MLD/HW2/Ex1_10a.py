import random

random.seed(69)

odds = 0.5
sampleSize = 1000

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

c1 = coins[0]
crand = coins[int(random.random()*sampleSize)]
cmini = val

print(c1/10)
print(crand/10)
print(cmini/10)