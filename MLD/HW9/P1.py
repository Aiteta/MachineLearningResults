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
	out = 2*((out)**0.5/16-0.75)/0.5/0.85/1.006/1.030
	return out

def symmetVal(data):
	ver = 0
	hor = 0
	for i in range(16):
		col = 0
		row = 0
		for j in range(16):
			col += data[i+16*j]
			row += data[j+16*i]
		ver += (col)**2
		hor += (row)**2
	out = (ver**0.5)/(hor**0.5)
	out = (math.log(out,16)-0.02)/0.068
	return out


digitFile = open("all.txt")
digitLines = digitFile.readlines()
digitFile.close()
writeFile = open("converted.txt","w")

i = 0
for line in digitLines:
	print(i)
	(digit, data) = convert(line)
	writeFile.write("{:d} {:.8f} {:.8f}\n".format(digit,intenVal(data),symmetVal(data)))
	if (digit == 1):
		plt.plot(intenVal(data),symmetVal(data),"ob")
	else:
		plt.plot(intenVal(data),symmetVal(data),"xr")
	i += 1
writeFile.close()

plt.show()



# ones  = numbers[1].copy()
# fives = numbers[5].copy()

# intensity1 = list()
# symmetry1  = list()

# for i in ones:
# 	plt.plot(intenVal(i),symmetVal(i), 'ob')
# for i in fives:
# 	plt.plot(intenVal(i),symmetVal(i), 'xr')

# plt.xlabel('Average Intensity')
# plt.ylabel('Vertical-Horizontal Ratio')
# plt.title('Feature Plot for digits 1 and 5')

# plt.show()

