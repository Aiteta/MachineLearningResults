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

# Read in in-sample and augmented-error results
f = open("outError2bEin.txt")
h = open("outError2bEaug.txt")
errorIn = f.readlines()
errorAug = h.readlines()

Ein = list()
Eaug = list()
Iteration = list()
for i in range(len(errorIn)):
  Ein.append(float(errorIn[i].split()[0]))
  Eaug.append(float(errorAug[i].split()[0]))
  Iteration.append(i+1)

f.close()
h.close()
plt.title("2-NN: Ein and Eaug vs. Iteration (Weight Decay)")
plt.loglog(Iteration,Ein,label="Ein")
plt.loglog(Iteration,Eaug,label="Eaug")
plt.legend()
plt.ylabel("Error")
plt.xlabel("Number of Iterations")

plt.show()
# Weights copied from output txt file
W_all = list()
W_all.append(np.array([[-3.149608807934686,-1.8093908252458766,0.6172661340613,-1.6308886387930404,6.5752792358991385,1.1804071486801624,0.11697643382377518,9.710635718416341,-1.3300527162622098,-0.13449365982190495],[4.9015944096713735,-0.8600316918596208,0.026882896368121068,2.3132410191990442,-10.939412130841772,-0.37171394571232264,0.10005584528918317,-16.60301296874509,1.8338329079732423,-0.0823595659149419],[8.01160174495725,2.7846650753832645,0.0735583410959674,6.4649209156759415,-16.30326953109963,-2.0740452164469176,0.13192413027748548,-26.27426984907526,7.225739070987673,-0.05563269895643485]]))
W_all.append(np.array([[-1.2569591135707234],[2.9735601627646786],[-2.7839816157510624],[-0.7851469702998434],[-3.7250550557495727],[4.595676825917467],[-1.347696017847679],[-0.21921580413931818],[-4.1424821111152585],[1.9541105513494363],[0.2934339266575347]]))
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
trainFile.close()

plt.title("2-NN: Decision Boundary (Weight Decay)")
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






