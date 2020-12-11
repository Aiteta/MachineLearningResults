import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

def convert(digitLine):
	digit = list(map(lambda x: -(float(x)/2+0.5)+1,digitLine.split()))
	data  = int(2*(-(digit.pop(0)-1)-0.5))
	return data, digit

def printer(data):
	m = np.zeros((16,16))
	for x in range(16):
		for y in range(16):
			m[x][y] = data[y+x*16]
	plt.matshow(m, cmap=plt.get_cmap('gray'), vmin=0.0, vmax=1.0)
	plt.show()
	i = input()


digitFile = open("train.txt")
digitLines = digitFile.readlines()
numbers = dict()

i = 0
for line in digitLines:
	print(i)
	(digit, data) = convert(line)
	if (not numbers.get(digit)):
		numbers[digit] = list()
		printer(data)

	numbers[digit].append(data)
	i += 1
	