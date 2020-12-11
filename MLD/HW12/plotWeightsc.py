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

# Read in in-sample and test error results
f = open("outError2cEin.txt",'r')
h = open("outError2cEtest.txt",'r')
errorIn = f.readlines()
errorTest = h.readlines()

Ein = list()
Etest = list()
Iteration = list()
for i in range(len(errorIn)):
  Ein.append(float(errorIn[i].split()[0]))
  Etest.append(float(errorTest[i].split()[0]))
  Iteration.append(i+1)

f.close()
h.close()
plt.title("2-NN: Ein and Etest vs. Iteration (Early Stopping)")
plt.loglog(Iteration,Ein,label="Ein")
plt.loglog(Iteration,Etest,label="Etest")
plt.legend()
plt.ylabel("Error")
plt.xlabel("Number of Iterations")

plt.show()

# Weights copied from output txt file
W_all = list()
W_all.append(np.array([[-1.625216311114576,3.756105993783424,8.597122249800345,0.06573590350419954,-5.635588742312383,-3.738705496641599,1.6621985970235,2.7326269112157404,0.44489961507913106,0.2964388891139071],[-1.4618923570784208,2.0807472053134792,-14.898144069530966,0.19342661669377162,9.443193549664647,5.942500732054514,-0.9242346494993973,-3.8748189529889636,0.7154590731545932,0.29403975094201684],[2.2862686175459013,-7.815378924935168,-25.197711932665573,-0.15669806386879184,14.240632988130502,7.860969175409071,-0.6952907086292589,-4.387059660210442,-1.2778824764171433,-0.2558847152284512]]))
W_all.append(np.array([[-1.8986189938360163], [1.4266441127958436], [4.116839939203112], [-3.4411722058898153], [-0.26824803701255867], [-4.448862303569502], [3.4539809887286554], [-1.4328078596995808], [1.8807251916526888], [0.5821231719179989], [-0.7188931616107062]]))
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

plt.title("2-NN: Decision Boundary (Early Stopping)")
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