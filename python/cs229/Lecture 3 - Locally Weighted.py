# /*
#  * @Author: Yuddi 
#  * @Date: 2021-08-24 17:42:14 
#  * @Last Modified by:    
#  * @Last Modified time: 2021-08-24 17:42:14 
#  */


from numpy.core.fromnumeric import choose
from numpy.random import rand
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np 
import math


N_SAMPLES = 500
TEST_SIEZ = 0.2
lr = 0.01

# make dummy regression data
noise = np.random.normal(0,2,size = N_SAMPLES).reshape(N_SAMPLES,1)
x = np.linspace(-5,5,N_SAMPLES).reshape(N_SAMPLES,1)
y = x**3 + noise
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIEZ, random_state=0)

# x:N_SAMPLES × N_FEATURES  y:N_SAMPLES × 1
# x_test:(N_SAMPLES*TEST_SIEZ) × N_FEATURES

#add x_0 into train and test data
x_0_train = np.ones((x_train.shape[0],1), dtype=x_train.dtype)
x_0_test = np.ones((x_test.shape[0],1), dtype=x_test.dtype)
x_train = np.concatenate((x_0_train,x_train),axis=1)
x_test = np.concatenate((x_0_test,x_test),axis=1)

theta = np.random.rand(2,1)
tau = 0.05

def hypotheses(theta,x):
    return np.matmul(theta.T,x)

def get_weight(x,x_point,tau):
    return math.exp(-((x[1,0]-x_point)**2)/((2*tau)**2))

def loss_function(h,y,w):
    return w*((y-h)**2)

def Stochastic_update_weights(theta,h,y,x,w):
    k = lr*w*(h-y)*(x)
    return theta- k

def Batch_update_weights(theta,batch,w):
    return theta- lr*batch


plt.ion()
for i in range(len(x_test)-90):
    x_point = x_test[i,1]
    loss = 0
    for epoch in range(20):
        for x,y in zip(x_train,y_train):
            x = x.reshape(theta.shape)
            h = hypotheses(theta,x)
            w = get_weight(x,x_point,tau)
            loss = loss_function(h,y,w)
            theta = Stochastic_update_weights(theta,h,y,x,w)
        print("[{0}/{1}], loss: {2}".format(epoch, 20, loss[0]))
        plt.cla()
        plt.scatter(x_train[:, 1], y_train, c='gray', marker='o', edgecolors='white')
        plt.ylim((-100, 100))
        plt.xlim((-7.5, 7.5))
        p_x = np.linspace(x_point-3, x_point+3, 100)
        plt.scatter(x_test[:,1],y_test,marker='o',c='orange')
        plt.scatter(x_point,y_test[i],marker='x',c='red')
        plt.plot(p_x, theta[1, 0] * p_x + theta[0, 0], c='orange')
        plt.pause(0.1)

plt.ioff()
plt.show()