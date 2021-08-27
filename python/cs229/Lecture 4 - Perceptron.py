# /*
#  * @Author: Yuddi 
#  * @Date: 2021-08-26 11:38:56 
#  * @Last Modified by:    
#  * @Last Modified time: 2021-08-26 11:38:56 
#  */



from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from math import exp, log


N_FEATURES = 2
N_SAMPLES = 500
TEST_SIEZ = 0.2

lr = 0.075

x, y = make_blobs(n_samples=N_SAMPLES, centers=2, n_features=N_FEATURES, random_state=3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIEZ, random_state=0)
# plt.scatter(x[:,0], x[:, 1], c=y, edgecolors='white', marker='s')
# plt.show()

x_0_train = np.ones((x_train.shape[0],1), dtype=x_train.dtype)
x_0_test = np.ones((x_test.shape[0],1), dtype=x_test.dtype)
x_train = np.concatenate((x_0_train,x_train),axis=1)
x_test = np.concatenate((x_0_test,x_test),axis=1)

THETA = np.random.rand(N_FEATURES+1).reshape(N_FEATURES+1,1)
H_train = np.zeros([y_train.shape[0], 1], dtype=y_train.dtype)

## Second step: Making hypothesis
def sigmoid_function(z):
	
	g = 1 / (1 + exp(-z))

	return g

def perceptron(z):
    if(z>=0):
        return 1
    else:
        return 0

def hypothesis(x, THETA):
	
	hypothesis = np.matmul(THETA.T, x)
	hypothesis = perceptron(hypothesis[0])
	return hypothesis



## Third step: Define a loss function
def compute_loss(X, Y, THETA):
	
	loss = 0
	for x, y in zip(X, Y):
		h_x = hypothesis(x, THETA)
		# if h_x == 1 --> log(1-1) --> error
		if h_x == 1:
			loss += (-y) *(log(h_x))
		else:
			loss +=  -(1-y) *log(1-h_x)

	return loss/(X.shape[0])

## Fourth step: Updating parameters
def update_parameters(THETA, LR, y, h_x, x):
	
	x = np.reshape(x, THETA.shape)
	THETA = THETA + LR *(y - h_x) * x


	return THETA


plt.figure(0)
plt.ion()
EPOCH = 2
for epoch in range(EPOCH):
	i = 0 # retrieve H_x
	for x, y in zip(x_train, y_train):
		# loss = compute_loss(x_train, y_train, THETA)
		# print('[{0}/{1}] loss is: {2}'.format(epoch+1, EPOCH, loss))
		H_train[i] = hypothesis(x, THETA)
		# print(H_train[i])
		THETA = update_parameters(THETA, lr, y, H_train[i], x)

		plt.cla()
		plt.scatter(x_train[:, 1], x_train[:, 2], c=H_train[:, 0], edgecolors='white', marker='s')
		plt.pause(0.001)
		i+=1
plt.ioff()
plt.show()

h_test = np.zeros(y_test.shape[0], dtype=y_test.dtype)
i = 0
for x,y in zip(x_test,y_test):
    h_test[i] = hypothesis(x,THETA)
    i+=1


plt.figure(1)
plt.scatter(x_test[:, 1], x_test[:, 2], c=h_test, edgecolors='white', marker='s')
plt.figure(2)
plt.scatter(x_test[:, 1], x_test[:, 2], c=y_test, edgecolors='white', marker='s')

plt.show()
