---
title: CS229_Lecture2_Linear Regression and Gradient Descent
date: 2021-08-21 11:15:46
tags: Machine Learning
categories: Python
mathjax: true
cover: /img/cs229/lecture2/cover.png
description: This course mainly explains Linear Regression and Gradient Descent including Batch Gradient Descent and Stochastic Gradient Descent. Especially, we use python to implementate the algorithm.
---





# LECTURE 2 - Linear Regression and Gradient Descent


## Linear Regression


&nbsp;&nbsp;There is a simple linear function below:
$$ h_{\theta}(x) = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + ... $$
&nbsp;&nbsp;In this function, $\theta_{i}$is the parameter(weight) of linear functions mapping from x to y. x is the input and y is the output. At the same time, we can simplify the equation like this:
$$ h(x) = \sum_{i=0}^{d}\theta_{i}x_{i} = \theta^{T}x$$
&nbsp;&nbsp;In this function, d is the number of features and we set $x_{0} = 1$. In order to make y close to $h_{x}$, we define a **cost function** to get the parameter of $\theta$:
$$ J(\theta) = \frac{1}{2}\sum_{i=1}^{n}(h_{\theta}(x^{(i)})-y^{(i)})^2$$
&nbsp;&nbsp;n is the number of training examples. We must choose $\theta$ to minimize $J(\theta)$. 
<br>

## Gradient Descent

&nbsp;&nbsp;Gradient Descent starts with some initial $\theta$, and repeated performs the update:
$$\theta_{j} = \theta_{j} - \alpha\frac{\partial}{\partial\theta_{j}}J(\theta)$$
This update is simultaneously performed on all values of j = 0,....,d. $\alpha$ is the **learning rate**. This Gradient Descent algorithm repeatedly takes a step in the direction of steepest decrease of $J$. Above, we have known the definition of $J(\theta)$.We have:
$$ \frac{\partial}{\partial\theta_{j}}J(\theta) = \frac{\partial}{\partial\theta_{j}}\frac{1}{2}(h_{\theta}(x)-y)^2 = (h_{\theta}(x)-y)x_{j}$$
So we can know that:
$$\theta_{j} = \theta_{j} - \alpha(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)}$$
<br>

### Batch Gradient Descent and Stochastic Gradient Descent
Batch Gradient Descent uses **every example** in the entire training set **on every step**. It runs like following algorithm:

>Repeat until convergence{
>
>$$ \theta_{j} = \theta_{j} -\alpha\sum_{i=1}^{n}(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)},(j=0,1,2...features)$$
>}

Stochastic Gradient Descent uses **one example** int the entire training set **on every step**. It runs like following algorithm:

>Loop{
>&nbsp;&nbsp;for i = 1 to n(or < n),{
>$$ \theta_{j} = \theta_{j} -\alpha(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)},(j=0,1,2...features)$$
>&nbsp;&nbsp;}
>}

When the training set is large, stochastic gradient descent is often preferred over batch gradient descent.

```python
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


```

## Normal Equations
$X$ is a matric and $\vec{y}$ is a vector whic contains all training data.
$$ J(\theta) = \frac{1}{2}\sum_{i=1}^{n}(h_{\theta}(x^{(i)})-y^{(i)})^2 = \frac{1}{2}(X\theta-\vec y)^{T}(X\theta-\vec y)$$
To minimise $J$, we must make$\nabla_{\theta}J(\theta) = 0$.
$$\begin{split}
\nabla_{\theta}J(\theta) &= \nabla_{\theta}\frac{1}{2}(X\theta-\vec y)^{T}(X\theta-\vec y) \\ 
&= \frac{1}{2}\nabla_{\theta}((X\theta)^{T}X\theta-(X\theta)^{T}\vec{y} - \vec{y}^{T}X\theta+\vec{y}^{T}\vec{y}) \\
&= \frac{1}{2}\nabla_{\theta}(\theta^{T}X^{T}X\theta-2(X^{T}\vec{y})^{T}\theta) \\
&= \frac{1}{2}(2X^{T}X\theta-2X^{T}\vec{y}) \\
&= X^{T}X\theta-X^{T}\vec{y} 
\end{split}$$

$$*.\nabla_{x} b^{T}x = b \qquad \nabla_{x}x^{T}Ax = 2Ax(if\ A\ symmetric) $$

We can obtain the normal equation:$X^{T}X\theta=X^{T}\vec{y}$. Thus, the value of $\theta$ is $(X^{T}X)^{-1}X^{T}\vec{y}$.

```python
# /*
#  * @Author: Yuddi 
#  * @Date: 2021-08-20 20:07:00 
#  * @Last Modified by:    
#  * @Last Modified time: 2021-08-20 20:07:00 
#  */

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np 
from numpy.linalg import inv

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

#Normal Equation
weights = np.matmul(inv(np.matmul(x_train.T,x_train)),np.matmul(x_train.T,y_train)).reshape(N_FEATURES+1,1)

# test
plt.scatter(x_test[:, 1], y_test, c='red', marker='o', edgecolors='white')
plt.plot(x_test, weights[1, 0] * x_test + weights[0, 0], c='orange')
plt.ylim((-25, 125))
plt.xlim((-3, 3))
plt.show()
```