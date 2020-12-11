import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math

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

def intenVal(data):
	out = 0
	for i in data:
		out += i**2
	out = (out)**0.5/16
	return out

def symmetVal(data):
	ver = 0
	hor = 0
	for i in range(16):
		col = 0
		row = 0
		for j in range(16):
			col += data[i+16*j]**2
			row += data[j+16*i]**2
		ver += (col)**0.5
		hor += (row)**0.5
	out = ver/hor
	out = math.log(out,16)
	return out


digitFile = open("train.txt")
digitLines = digitFile.readlines()
numbers = dict()

i = 0
for line in digitLines:
	print(i)
	(digit, data) = convert(line)
	if (not numbers.get(digit)):
		numbers[digit] = list()

	numbers[digit].append(data)
	i += 1

ones  = numbers[1].copy()
fives = numbers[5].copy()

intensity1 = list()
symmetry1  = list()

for i in ones:
	plt.plot(intenVal(i),symmetVal(i), 'ob')
for i in fives:
	plt.plot(intenVal(i),symmetVal(i), 'xr')

plt.xlabel('Average Intensity')
plt.ylabel('Vertical-Horizontal Ratio')
plt.title('Feature Plot for digits 1 and 5')

plt.show()

