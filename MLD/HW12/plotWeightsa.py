import random
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math
import time

# Performs forward propogation
# Output of each Layer is recorded and output
def forwardProp(X_all,W_all,L,out):
	X = list()
	tanh_v = np.vectorize(math.tanh)
	x = np.array(X_all).T
	# Iterate through all Layers of NN,
	# All operations are performed on the entire data set at a times
	for i in range(L):
		x = np.insert(x,0,[1],axis = 0)
		X.append(x)
		x = np.dot(np.transpose(W_all[i]),x)
		if (i != L-1):
			x = tanh_v(x)
	X.append(out(x))
	return X

# Read in in-sample error results
f = open("outError2a.txt")
errorLines = f.readlines()

Ein = list()
Iteration = list()
j = 1
last = -1
for i in errorLines:
	last = i.split()[0]
	Ein.append(float(last))
	Iteration.append(j)
	j += 1
f.close()
plt.title("2-NN: Ein vs. Iteration (Variable Learning Gradient)")
plt.loglog(Iteration,Ein)
plt.ylabel("In-Sample Error")
plt.xlabel("Number of Iterations")

plt.show()

# Weights copied from output txt file
W_all = list()
W_all.append(np.array([[-9.8753099388616,1.7875288896108952,-4.018757848287952,-0.17382803320512805,1.701767321283119,-6.305729580185575,-2.7210228760295254,-4.517124841380812,-0.23493426982830362,-2.6977971605355453],[16.799273721446053,-0.8451258595726058,6.143307470799612,-0.05142831133313521,-0.8048635989420189,10.315736054957927,-2.6115668695338656,-2.1025164644483065,-0.1473545986128286,3.4030537557955207],[28.24266409063662,-1.005889264267597,8.529738708210399,-0.09508533329564271,-0.9671619336912118,15.460070089057213,2.642169363778177,9.167221323016099,-0.21840578962893892,4.393786460047013]]))
W_all.append(np.array([[-1.97449652185618],[3.7142811502791817],[-1.1929705217053364],[3.76107440881742],[0.21552808834179807],[-1.0300351061643458],[-4.971610228884467],[1.441193637142328],[-4.816733847686682],[0.46012354007754536],[-2.3512071198640254]]))
# Plot Decision Boundary
for i in range(51):
	for j in range(51):
		x = [i/25-1,j/25-1]
		X = forwardProp(x,W_all,2,np.vectorize(np.sign))
		y = X[-1]
		if (y < 0):
			plt.plot(i/25-1,j/25-1,'sr', alpha = 0.2)
		else:
			plt.plot(i/25-1,j/25-1,'sb', alpha = 0.2)


# Plot training points
trainFile = open("train.txt")
trainLines = trainFile.readlines()

for i in trainLines:
	data = i.split()
	if (int(data[0]) == 1):
		plt.plot(float(data[1]),float(data[2]),'ob')
	else:
		plt.plot(float(data[1]),float(data[2]),'xr')

plt.title("2-NN: Decision Boundary (Variable Learning Gradient)")
plt.xlabel("Average Intensity")
plt.ylabel("Vertical-Horizontal Ratio")
plt.show()

# Calculate Out of sample error
testFile = open("test.txt")
testLines = testFile.readlines()

Etest = 0
for i in testLines:
  data = i.split()
  x = [float(data[1]),float(data[2])]
  y = forwardProp(x,W_all,2,np.vectorize(np.sign))[-1]
  ytrue = 0
  if (int(data[0]) == 1):
    ytrue = 1
  else:
    ytrue = -1
  if (np.sign(y) != ytrue):
    Etest += 2
Etest /= len(testLines)
print(Etest)
    
testFile.close()