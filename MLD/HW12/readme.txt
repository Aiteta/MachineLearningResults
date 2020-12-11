All code was run on Python using the following libraries

random
matplotlib.pyplot
matplotlib.colors
numpy
math
time
qpsolvers.solve_qp

All files are also submitted which contain the below code and for plotting the found weights, support vectors and biases. Additionally txt files were created for the output of the nueral network due to the fear of losing that data or possibly formatting it wrong on plots, such data is several megabytes large but can be sent over if requested. Lastly plotting of weight decision boundaries and error vs iteration plots are done in separate files which are also included. If there are any questions or concerns please ask me.

# Derivative of tanh function
def tanh_prime(x):
	return 1-x**2

# Identity function
def identity(s):
	return s

# Derivative of Identity Function
def identity_prime(x):
	return 1

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

# Performed backward propogation
# Delta of each hidden layer and output layer is returned
# Which is used in calculating the gradient
def backwardProp(X,W_all,Y_all,out_prime,L):
	Delta = list()
	# Delta of output
	Delta.append(2*(X[-1]-Y_all)*out_prime(X[-1]))
	# Iteration through all hidden layers
	for i in range(L-1):
		l = len(X)-2-i
		x = X[l][1:,:]
		theta = np.vectorize(tanh_prime)(x)
		temp = np.dot(W_all[l][1:,:],Delta[-1])
		Delta.append(np.multiply(temp,theta))
	return Delta

# In-Sample Error is calculated
# Gradient of each weight matrix is returned
def E_in(X_all,Y_all,W_all,L,out,out_prime):
	Ein = 0
	G_all = list(map(lambda w: np.zeros(w.shape),W_all))
	N = len(X_all)
	# Forward Propogation
	X = forwardProp(X_all,W_all,L,out)
	# Backward Propogation
	Delta = backwardProp(X,W_all,Y_all,out_prime,L)
	error_v = np.vectorize(lambda x: x**2/N)
	Ein = sum(error_v(X[-1]-Y_all)[0])
	# Calculation of gradient
	for l in range(L):
		Xl = X[l].T
		Dl = Delta[L-1-l].T
		G_all[l] = sum(list(map(lambda x,d: np.outer(x,d)/N,Xl,Dl)))
	return (Ein,G_all)

# Weight Decay with use of parametrization parameter lambda
# Return final Weights, final in-sample error 
# and the in-sample error over iterations
# In addition, Augmented Error
def WeightDecay(X,Y,W_all,L,numIt,lamb):
	# Initial Setup
	alpha = 1.08
	beta = 0.7
	eta = 0.5
	Ein = 0
	Eaug = 0
	new = 0
	Ein_all = list()
	Eaug_all = list()
	identity_v = np.vectorize(identity)
	identity_prime_v = np.vectorize(identity_prime)
	N = len(X)
	# Iterate over number of specified iterations
	for i in range(numIt):
		# Calculate In-Sample error and gradient
		(Ein, G_all) = E_in(X,Y,W_all,L,identity_v,identity_prime_v)
		Ein_all.append(Ein)
		# Calculate Augmented Error
		Eaug = Ein
		for i in range(len(W_all)):
			Eaug += lamb/N*np.sum(np.vectorize(lambda x: x**2)(W_all[i]))
		# Calculate new weights and check 
		# if the new weights perform better or worse
		# Based on Augmented Error
		Eaug_all.append(Eaug)
		temp = list(map(lambda w,g: w-eta*(g+2*lamb/N*w),W_all,G_all))
		new = E_in_noGrad(X,Y,temp,L,identity_v,identity_prime_v)
		newaug = new
		for i in range(len(W_all)):
			newaug += lamb/N*np.sum(np.vectorize(lambda x: x**2)(temp[i]))
		if (Eaug > newaug):
			W_all = temp
			eta = alpha*eta
		else:
			eta = beta*eta
	return (W_all,Ein,Ein_all,Eaug,Eaug_all)

# Nueral network, perform learning on given dataset over predefined
# Iterations and user-defined hidden layers
# Return final weights and in-sample error analysis
def NueralNetwork(X,Y,L,numIt,lamb):
	# Initialize weights
	W_all = list()
	for i in range(L):
		W = list()
		for j in range(d[i+1]):
			w = list()
			for k in range(d[i]+1):
				w.append(0.2*(random.random()-0.5))
			W.append(w)
		W = np.array(W)
		W_all.append(np.transpose(W))
	# LEARN
	(W_all,Ein,Ein_all,Eaug,Eaug_all) = WeightDecay(X,Y,W_all,L,numIt,lamb)
	return (W_all,Ein,Ein_all,Eaug,Eaug_all)

# Variable Gradient Descent
# Return final Weights, final in-sample error 
# and the in-sample error over iterations
def VarGradDes(X,Y,W_all,L,numIt):
	# Initial Setup
	alpha = 1.075
	beta = 0.65
	eta = 0.5
	Ein = 0
	new = 0
	Ein_all = list()
	identity_v = np.vectorize(identity)
	identity_prime_v = np.vectorize(identity_prime)
	# Iterate over number of specified iterations
	for i in range(numIt):
		# Calculate In-Sample error and gradient
		(Ein, G_all) = E_in(X,Y,W_all,L,identity_v,identity_prime_v)
		Ein_all.append(Ein)
		# Calculate new weights and check 
		# if the new weights perform better or worse
		temp = list(map(lambda w,g: w-eta*g,W_all,G_all))
		new = E_in_noGrad(X,Y,temp,L,identity_v,identity_prime_v)
		if (Ein > new):
			W_all = temp
			eta = alpha*eta
		else:
			eta = beta*eta
	return (W_all,Ein,Ein_all)

# Nueral network, perform learning on given dataset over predefined
# Iterations and user-defined hidden layers
# Return final weights and in-sample error analysis
def NueralNetwork(X,Y,L,d,numIt):
	# Initialize weights
	W_all = list()
	for i in range(L):
		W = list()
		for j in range(d[i+1]):
			w = list()
			for k in range(d[i]+1):
				w.append(0.2*(random.random()-0.5))
			W.append(w)
		W = np.array(W)
		W_all.append(np.transpose(W))
	# LEARN
	(W_all,Ein,Ein_all) = VarGradDes(X,Y,W_all,L,numIt)
	return (W_all,Ein,Ein_all)

# Early Stopping
# Return final Weights, final in-sample error 
# and the in-sample error over iterations
# In addition, Test Error
# Final Weights return are based on best test error performance
def EarlyStopping(X,Y,W_all,L,numIt,size):
	# Initial Setup
	alpha = 1.075
	beta = 0.65
	eta = 0.5
	Ein = 0
	new = 0
	# Seperate Data into Validation Set and Training Set
	testX = list()
	testY = list()
	newX = list()
	newY = list()
	test_it = set(random.sample(range(len(X)),size))
	for i in range(len(X)):
		if (i in test_it):
			testX.append(X[i])
			testY.append(Y[i])
		else:
			newX.append(X[i])
			newY.append(Y[i])
	X = np.array(newX)
	Y = np.array(newY)
	testX = np.array(testX)
	testY = np.array(testY)
	# More initial setup
	Ein_all = list()
	Etest_all = list()
	bestEtest = -1
	bestW = W_all
	identity_v = np.vectorize(identity)
	identity_prime_v = np.vectorize(identity_prime)
	# Iterate over number of specified iterations
	for i in range(numIt):
		# print(Ein)
		# print(bestEtest)
		if (i %100 == 0):
			print(i)
		# 
		(Ein, G_all) = E_in(X,Y,W_all,L,identity_v,identity_prime_v)
		Etest = E_in_noGrad(testX,testY,W_all,L,identity_v,identity_prime_v)
		Ein_all.append(Ein)
		Etest_all.append(Etest)
		# Save weights if they perform the best on test error
		if (bestEtest == -1 or Etest < bestEtest):
			bestEtest = Etest
			bestW = W_all
		# Calculate new weights and check 
		# if the new weights perform better or worse
		temp = list(map(lambda w,g: w-eta*g,W_all,G_all))
		new = E_in_noGrad(X,Y,temp,L,identity_v,identity_prime_v)
		if (Ein > new):
			W_all = temp
			eta = alpha*eta
		else:
			eta = beta*eta
	return (bestW,Ein,Ein_all,bestEtest,Etest_all)

# Nueral network, perform learning on given dataset over predefined
# Iterations and user-defined hidden layers
# Return final weights and in-sample error analysis
def NueralNetwork(X,Y,L,numIt,size):
	# Initialize Weights
	W_all = list()
	for i in range(L):
		W = list()
		for j in range(d[i+1]):
			w = list()
			for k in range(d[i]+1):
				w.append(0.2*(random.random()-0.5))
			W.append(w)
		W = np.array(W)
		W_all.append(np.transpose(W))
	# LEARN
	(bestW,Ein,Ein_all,bestEtest,Etest_all) = EarlyStopping(X,Y,W_all,L,numIt,size)
	return (bestW,Ein,Ein_all,bestEtest,Etest_all)

This was not done inside a function
# Organize Data for QP solve
N = len(Y)
P = np.array(QD)    # QD
q = np.ones(N)*-1   # p
G = np.zeros((N,N)) # -Ad(aplha) <= c wasn't working
h = np.zeros(N)     # So used Ad(alpha) = c, specifying a lower
A = np.array(Y)     # AD, and upper bound for alpha
b = np.zeros(1)     # c
lower = np.zeros(N) # Lower Bound
upper = np.ones(N)*C# Upper Bound
alpha = solve_qp(P,q,G,h,A,b,lower,upper)
# Find Support Vectors
sup_vec_index = list()
for i in range(len(alpha)):
	if (alpha[i] > tol):
		sup_vec_index.append(i)
# Calculate Bias
b = Y[sup_vec_index[0]]
for i in range(len(sup_vec_index)):
	j = sup_vec_index[i]
	b -= Y[j]*alpha[j]*K(X[j],X[sup_vec_index[0]])

# Organize Data for QP solve except QD and p
N = len(Y)
G = np.zeros((N,N)) # -Ad(aplha) <= c wasn't working
h = np.zeros(N)     # So used Ad(alpha) = c, specifying a lower
A = np.array(Y)     # AD, and upper bound for alpha
b = np.zeros(1)     # c
lower = np.zeros(N) # Lower Bound
upper = np.ones(N)*C# Upper Bound
lower = np.zeros(N-1)
upper = np.ones(N-1)*C
Ecv_all = list()
C_all = list()
# Perform Ecv over 10^-2 <= C < 30
# Perform 10-fold Ecv
# Save Ecv for each removed data point
for i in range(31):
	C = 10**(i/30*3.5-2)
	C_all.append(C)
	Ecv = 0
	for j in range(len(X)//10):
		x = X.pop(0)
		y = Y.pop(0)
		QD = list()
		for i in range(len(X)):
			w = list()
			for j in range(len(X)):
				# print(np.dot(X[i],X[j]))
				if (i != j):
					w.append(Y[i]*Y[j]*K(X[i],X[j]))
				else:
					w.append(Y[i]*Y[j]*K(X[i],X[j])+10**(-8))
			QD.append(w)
		# Create QD and p and do QP solve
		P = np.array(QD) # QD
		A = np.array(Y)  # p
		alpha = solve_qp(P,q,G,h,A,b,lower,upper)
		# Find Support Vectors
		sup_vec_index = list()
		for i in range(len(alpha)):
			if (alpha[i] > tol):
				sup_vec_index.append(i)
		# Calculate Bias
		d = Y[sup_vec_index[0]]
		for i in range(len(sup_vec_index)):
			j = sup_vec_index[i]
			d -= Y[j]*alpha[j]*K(X[j],X[sup_vec_index[0]])
		# Classify removed 
		guess = 0+d
		for k in range(len(sup_vec_index)):
			l = sup_vec_index[k]
			guess += Y[l]*alpha[l]*K(X[l],np.array([i/25-1,j/25-1]))
		if (np.sign(guess) != y):
			Ecv += 2
		X.append(x)
		Y.append(y)
	Ecv_all.append(Ecv/N*10)
