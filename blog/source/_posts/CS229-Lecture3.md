---
title: CS229_Lecture3_Locally Weighted and Logistic Regression
date: 2021-08-25 19:23:33
tags: Machine Learning
categories: Python
mathjax: true
cover: /img/cs229/lecture3/cover.png
description: This course mainly explains Locally Weighted and Logistic Regression. Especially, we use python to implementate the algorithm.

---


# Lecture 3 - Locally Weighted and Logistic Regression

## Locally Weighted Linear Regression

&nbsp;&nbsp;The locally weighted linear regression algorithm does something different from the Linear Regression. Its cost function is:
$$J_{\theta} = \frac{1}{2}\sum_{i=1}^{n}w^{(i)}(y^{(i)}-\theta^{T}x^{(i)})^{2}$$
In this function, the definition of $w^{(i)}$ is:
$$w^{(i)} = exp(-\frac{(x^{(i)}-x)^{2}}{2\tau^{2}})$$
&nbsp;&nbsp;We can find that $w^{i}$ is close to 1 when $|x^{i}-x|$ is small. Therefore, $\theta$ will be chosen to be something to do with $x$. And the parameter $\tau$, which is also called the **bandwidth** parameter, controls how quickly the weight of a training example falls off with distance of its $x^{(i)}$ from the query point $x$.
&nbsp;&nbsp;Linear Regression is a parametric algorithm, which has fixed parameters. And Locally Weighted Linear Regression is a non-parametric algorithm.


```python

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

```

## Logistic Regression

The hypothes of logistic regression is:
$$\begin{split}
    h_{\theta}(x) = &g(\theta^{T}x) = \frac{1}{1+e^{-\theta^{T}x}}  \\
    &g(z) = \frac{1}{1+e^{-z}}
\end{split}
$$
$g(z)$ is called the logistic function or the sigmoid function. At the same time, the stochastic gradient ascent rile is:
$$\theta_{j}:=\theta_{j}+\alpha(y^{(i)}-h_{\theta}(x^{(i)}))x_{j}^{(i)}$$
which is same with the linear regression gradient ascent rule.


```python
# /*
#  * @Author: Yuddi 
#  * @Date: 2021-08-25 11:58:49 
#  * @Last Modified by:    
#  * @Last Modified time: 2021-08-25 11:58:49 
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

def hypothesis(x, THETA):
	
	hypothesis = np.matmul(THETA.T, x)
	hypothesis = sigmoid_function(hypothesis[0])
	return hypothesis

## Third step: Define a loss function
def compute_loss(X, Y, THETA):
	
	loss = 0
	for x, y in zip(X, Y):
		h_x = hypothesis(x, THETA)
		# if h_x == 1 --> log(1-1) --> error
		if h_x == 1:
			h_x = 1-0.0000000000001
		loss += (-y) *(log(h_x) - (1-y) *log(1-h_x))

	return loss/(X.shape[0])

## Fourth step: Updating parameters
def update_parameters(THETA, LR, y, h_x, x):
	
	x = np.reshape(x, THETA.shape)
	THETA = THETA + LR *(y - h_x) * x


	return THETA


plt.figure(0)
plt.ion()
EPOCH = 4
for epoch in range(EPOCH):
	i = 0 # retrieve H_x
	for x, y in zip(x_train, y_train):
		loss = compute_loss(x_train, y_train, THETA)
		print('[{0}/{1}] loss is: {2}'.format(epoch+1, EPOCH, loss))
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



```