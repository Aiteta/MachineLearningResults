import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random

def add(w,x,y):
	for i in range(len(x)):
		w[i] += x[i]*y

def mult(x,w):
	out = 0
	for i in range(len(x)):
		out += x[i]*w[i]
	return out

def check(X,Y,w):
	start = int(random.random()*len(X)//1)
	# print(start)
	for i in range(start,len(X)):
		if (sign(mult(X[i],w)) != sign(Y[i])):
			return i
	for i in range(start):
		if (sign(mult(X[i],w)) != sign(Y[i])):
			return i
	return -1

def sign(x):
	if (x < 0):
		return -1
	elif (x > 0):
		return 1
	return 0

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
	return out

def errorCalc(X,Y,w):
	e_total = 0
	for i in range(len(X)):
		if (sign(mult(X[i],w)) != sign(Y[i])):
			e_total += 1
	print("Error",e_total)
	return e_total/len(X)



digitFile = open("train.txt")
digitLines = digitFile.readlines()
numbers = dict()

i = 0
for line in digitLines:
	(digit, data) = convert(line)
	if (not numbers.get(digit)):
		numbers[digit] = list()

	numbers[digit].append(data)
	i += 1

ones  = numbers[1].copy()
fives = numbers[5].copy()

X = list()
Y = list()

plt.figure(1)
for i in ones:
	inten = intenVal(i)
	symmet = symmetVal(i)
	X.append([1,inten,symmet,inten**2,inten*symmet,symmet**2,inten**3,(inten**2)*symmet,inten*(symmet**2),symmet**3])
	plt.plot(inten,symmet, 'ob')
	Y.append(1)
for i in fives:
	inten = intenVal(i)
	symmet = symmetVal(i)
	X.append([1,inten,symmet,inten**2,inten*symmet,symmet**2,inten**3,(inten**2)*symmet,inten*(symmet**2),symmet**3])
	plt.plot(inten,symmet, 'xr')
	Y.append(-1)

xw = np.linspace(0.8,0.95, 2)
w = [0]*10
minw = w.copy()
minerror = -1
for i in range(10000):
	print(i)
	index = check(X,Y,w)
	error = errorCalc(X,Y,w)
	if (error <= minerror or minerror == -1):
		minerror = error
		minw = w.copy()
	add(w,X[index],Y[index])

print(minw)
print(errorCalc(X,Y,minw))
print(minerror)

# yw = -minw[1]/minw[2]*xw-minw[0]/minw[2]
print(w)
# plt.plot(xw,yw,'g',label='Final Hypothesis')


for i in range(201):
	print(i)
	for j in range(201):
		inten = 0.65+i*(1.0-0.65)/200
		symmet = 0.88+j*(1.04-0.88)/200
		a = [1,inten,symmet,inten**2,inten*symmet,symmet**2,inten**3,(inten**2)*symmet,inten*(symmet**2),symmet**3]
		b = mult(a,minw)
		if (abs(b) < 0.1):
			plt.plot(inten,symmet,'g.')



plt.xlabel('Average Intensity')
plt.ylabel('Vertical-Horizontal Ratio')
plt.title('Feature Plot for digits 1 and 5: Training Set')

plt.show()

plt.figure(2)

digitFile = open("test.txt")
digitLines = digitFile.readlines()
numbers = dict()

i = 0
for line in digitLines:
	(digit, data) = convert(line)
	if (not numbers.get(digit)):
		numbers[digit] = list()

	numbers[digit].append(data)
	i += 1

ones  = numbers[1].copy()
fives = numbers[5].copy()

X = list()
Y = list()

plt.figure(2)
for i in ones:
	inten = intenVal(i)
	symmet = symmetVal(i)
	X.append([1,inten,symmet,inten**2,inten*symmet,symmet**2,inten**3,(inten**2)*symmet,inten*(symmet**2),symmet**3])
	plt.plot(inten,symmet, 'ob')
	Y.append(1)
for i in fives:
	inten = intenVal(i)
	symmet = symmetVal(i)
	X.append([1,inten,symmet,inten**2,inten*symmet,symmet**2,inten**3,(inten**2)*symmet,inten*(symmet**2),symmet**3])
	plt.plot(inten,symmet, 'xr')
	Y.append(-1)

# xw = np.linspace(0.8,0.95, 2)
# yw = -minw[1]/minw[2]*xw-minw[0]/minw[2]
print(errorCalc(X,Y,minw))
# plt.plot(xw,yw,'g',label='Final Hypothesis')

for i in range(201):
	print(i)
	for j in range(201):
		inten = 0.65+i*(1.0-0.65)/200
		symmet = 0.88+j*(1.04-0.88)/200
		a = [1,inten,symmet,inten**2,inten*symmet,symmet**2,inten**3,(inten**2)*symmet,inten*(symmet**2),symmet**3]
		b = mult(a,minw)
		if (abs(b) < 0.1):
			plt.plot(inten,symmet,'g.')



plt.xlabel('Average Intensity')
plt.ylabel('Vertical-Horizontal Ratio')
plt.title('Feature Plot for digits 1 and 5: Test Set')

plt.show()
