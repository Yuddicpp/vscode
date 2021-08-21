# /*
#  * @Author: Yuddi 
#  * @Date: 2021-08-20 14:43:51 
#  * @Last Modified by:    
#  * @Last Modified time: 2021-08-20 14:43:51 
#  */

from numpy.core.fromnumeric import choose
from numpy.random import rand
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np 

N_FEATURES = 1
N_SAMPLES = 500
TEST_SIEZ = 0.2

# make dummy regression data
x, y = datasets.make_regression(n_samples=N_SAMPLES, n_features=N_FEATURES, noise=20, random_state=0, bias=50)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIEZ, random_state=0)

# x:N_SAMPLES × N_FEATURES  y:N_SAMPLES × 1
# x_test:(N_SAMPLES*TEST_SIEZ) × N_FEATURES

#add x_0 into train and test data
x_0_train = np.ones((x_train.shape[0],1), dtype=x_train.dtype)
x_0_test = np.ones((x_test.shape[0],1), dtype=x_test.dtype)
x_train = np.concatenate((x_0_train,x_train),axis=1)
x_test = np.concatenate((x_0_test,x_test),axis=1)

# θ parameters
weights = np.random.rand(N_FEATURES+1).reshape(N_FEATURES+1,1)

def hypotheses(x,weights):
    return np.matmul(x,weights)

def loss_function(h,y):
    return 0.5*(h-y)**2

def Stochastic_update_weights(weights,h,y,x):
    k = lr*(h-y)*(x.T)
    return weights- k

def Batch_update_weights(weights,batch):
    return weights- lr*batch

x_line = np.linspace(-2.5, 2.5, 50)
plt.ion()


# Stochastic Gradient Descent
lr = 0.1
Epoch = 10
for i in range(Epoch):
    loss = 0
    for x,y in zip(x_train,y_train):
        x = x.reshape(1,len(x))
        h = hypotheses(x,weights)
        loss += loss_function(h,y)
        weights = Stochastic_update_weights(weights,h,y,x)
    plt.cla()
    plt.scatter(x_train[:, 1], y_train, c='purple', marker='o', edgecolors='white')
    plt.plot(x_line, weights[1, 0] * x_line + weights[0, 0], c='orange')
    plt.ylim((-25, 125))
    plt.xlim((-3, 3))
    plt.pause(0.01)
    print("Epoch:{0}, loss: {1}".format(i+1,loss[0,0]/x_train.shape[0]))

plt.ioff()
plt.show()

plt.ion()
# Batch Gradient Descent
lr = 0.001
Epoch = 10
for i in range(Epoch):
    batch = np.zeros(N_FEATURES+1).reshape(N_FEATURES+1,1)
    loss = 0
    for x,y in zip(x_train,y_train):
        x = x.reshape(1,len(x))
        h = hypotheses(x,weights)
        loss += loss_function(h,y)
        batch += (h-y)*(x.T)
    weights = Batch_update_weights(weights,batch)
    print("Epoch:{0}, loss: {1}".format(i+1,loss[0,0]/x_train.shape[0]))
    plt.cla()
    plt.scatter(x_train[:, 1], y_train, c='purple', marker='o', edgecolors='white')
    plt.plot(x_line, weights[1, 0] * x_line + weights[0, 0], c='orange')
    plt.ylim((-25, 125))
    plt.xlim((-3, 3))
    plt.pause(0.01)


plt.ioff()
plt.show()


# test
plt.scatter(x_test[:, 1], y_test, c='red', marker='o', edgecolors='white')
plt.plot(x_test, weights[1, 0] * x_test + weights[0, 0], c='orange')
plt.ylim((-25, 125))
plt.xlim((-3, 3))
plt.show()
