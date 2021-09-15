---
title: CS229_Lecture4_Perceptron and Generalized Linear Model
date: 2021-09-01 14:48:46
tags: Machine Learning
categories: Python
mathjax: true
cover: /img/cs229/lecture4/cover.png
description: This section will show the perceptron learning algorithm and  how models in the GLM family can be derived and applied to other classification and regression problems.

---


# Perceptron and Generalized Linear Model

## Perceptron
&nbsp;&nbsp;The perceptron learning algorithm aims to solve the problem of binary classification. And the function is:
$$g(z) = \begin{cases}
    \ 1, & z\geq0 \\
    \ 0, & z<0
\end{cases}$$
&nbsp;&nbsp;We let $h_{\theta} = g(\theta^{T}x)$ as Logistic Regression alogrithm but using this modified definition of $g(z)$, and we use the update rule
$$\theta_{j}:=\theta_{j} + \alpha(y^{(i)}-h_{\theta}(x^{(i)}))x_{j}^{(i)}$$

```python

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



```


## Generalized Linear Model

### The Exponential Family
&nbsp;&nbsp;The form of exponential family is:
$$p(y;\eta) = b(y)exp(\eta^{T}T(y)-a(\eta))$$
&nbsp;&nbsp;In this equation, $\eta$ is called the **natural parameter** of the distribution. $T(y)$ is the **sufficient statistic**(often $T(y) = y$). $a(\eta)$ is the **log partition function**.
&nbsp;&nbsp;For example, the Bernoulli distribution is one of exponential family distributions.
$$\begin{split}
    p(y;\phi) & =  \phi^{y}(1-\phi)^{1-y} \\
    & = exp(ylog\phi + (1-y)log(1-\phi)) \\
    & = exp(log(\frac{\phi}{1-\phi})y+log(1-\phi))
\end{split}$$
&nbsp;&nbsp;We can find 
$$\begin{split}
    b(y) & = 1 \\
    \eta^{T} & = log(\frac{\phi}{1-\phi}) \\
    a(\eta) & = -log(1-\phi) \\
    & = log(1+e^{\eta})
\end{split}$$
At the same time, the exponential family also includes Gaussian, Binomial, Multinomial, Exponential, Poisson, Dirichlet distribution.

### GLM
&nbsp;&nbsp;To derive a GLM for a problem, we need to make three assumptions about the conditional distribution of $y$ given $x$ and about our model:

1. $y|x;\theta$~ExponentialFamily($\eta$)
2. $h(x)=E[y|x]$
3. $\eta=\theta^{T}x$
  
#### Ordinary Least Squares
we model the conditional distribution of y given x as a Gaussian $N(\mu,\sigma^{2})$. According to GLM, we know that:
$$\begin{split}
    h_{\theta}(x) &= E[y|x;\theta] \\
    & = \mu \\
    & = \eta \\
    & = \theta^{T}x
\end{split}$$

#### Logistic Regression
we model the conditional distribution of y given x as a Bernoulli($\phi$). Also, we can know that:
$$\begin{split}
    h_{\theta}(x) &= E[y|x;\theta] \\
    & = \phi \\
    & = 1/(1+e^{-\eta}) \\
    & = 1/(1+e^{-\theta^{T}x}) 
\end{split}$$

#### Softmax Regression
&nbsp;&nbsp;To parameterize a multinomial over k possible outcomes, one could use $k$ parameters $\phi_{1},...,\phi_{k}$ specifying the probability of each of the outcomes. However, these parameters would be redundant, or more formally, they would not be independent (since knowing any $k-1$ of the $\phi_{i}'s$ uniquely determines the last one, as they must satisfy $\sum_{i=1}^{k}\phi_{i}=1$). So, we will instead parameterize the multinomial with only $k-1$ parameters,$\phi_{1},...,\phi_{k}-1$, where $\phi_{i} = p(y=i;\phi)$, and $p(y=k;\phi) = 1-\sum_{i=1}^{k-1}\phi_{i}$. For notational convenience, we will also let $\phi_{k} = 1-\sum_{i=1}^{k-1}\phi_{i}$, but we should keep in mind that this is not a parameter, and that it is fully specified by $\phi_{1},...,\phi_{k}-1$.
&nbsp;&nbsp;And we can know:
$$\begin{split}
    p(y;\phi) & = \phi_{1}^{1\{y=1\}}\phi_{2}^{1\{y=2\}}\phi_{3}^{1\{y=3\}}...\phi_{k}^{1\{y=k\}} \\
    & = \phi_{1}^{1\{y=1\}}\phi_{2}^{1\{y=2\}}\phi_{3}^{1\{y=3\}}...\phi_{k}^{1-\sum_{i=1}^{k-1}\{y=i\}} \\
    & = exp((T(y))_{1}log(\phi_{1})+(T(y))_{2}log(\phi_{2})+...+(1-\sum_{i=1}^{k-1}(T(y))_{i})log(\phi_{k}))
\end{split}$$
where


$$
\begin{aligned}
\eta &=\left[\begin{array}{c}
\log \left(\phi_{1} / \phi_{k}\right) \\
\log \left(\phi_{2} / \phi_{k}\right) \\
\vdots \\
\log \left(\phi_{k-1} / \phi_{k}\right)
\end{array}\right] \\
a(\eta) &=-\log \left(\phi_{k}\right) \\
b(y) &=1
\end{aligned}
$$
And hypothesis will be:
$$
\begin{aligned}
h_{\theta}(x) &=\mathrm{E}[T(y) \mid x ; \theta] \\
&=\mathrm{E}\left[\begin{array}{c}
1\{y=1\} \\
1\{y=2\} \\
\vdots \\
1\{y=k-1\}
\end{array} \mid x ; \theta\right] \\
&=\left[\begin{array}{c}
\phi_{1} \\
\phi_{2} \\
\vdots \\
\phi_{k-1}
\end{array}\right] \\
&=\left[\begin{array}{c}
\frac{\exp \left(\theta_{1}^{T} x\right)}{\sum_{j=1}^{k} \exp \left(\theta_{j}^{T} x\right)} \\
\frac{\exp \left(\theta_{2}^{T} x\right)}{\sum_{j=1}^{k} \exp \left(\theta_{j}^{T} x\right)} \\
\vdots \\
\frac{\exp \left(\theta_{k-1}^{T} x\right)}{\sum_{j=1}^{k} \exp \left(\theta_{j}^{T} x\right)}
\end{array}\right] .
\end{aligned}
$$