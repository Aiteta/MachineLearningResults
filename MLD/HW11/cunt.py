import random
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math
import time


w = np.array([0.30130715, -0.19601817, -1.44835951])

for i in range(51):
	for j in range(51):
		x = i/25-1
		y = j/25-1
		X = [1,x,y]
		if (np.dot(X,w) < 0):
			plt.plot(x,y,'sr', alpha = 0.1)
		else:
			plt.plot(x,y,'sb', alpha = 0.1)
plt.show()