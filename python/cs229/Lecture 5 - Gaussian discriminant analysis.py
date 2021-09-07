# /*
#  * @Author: Yuddi 
#  * @Date: 2021-09-03 16:00:54 
#  * @Last Modified by:    
#  * @Last Modified time: 2021-09-03 16:00:54 
#  */



from numpy.core.fromnumeric import ptp
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1)





x, y = make_blobs(n_samples=300, n_features=2, centers=2, random_state=12)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
# plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolors='white')
# plt.show()

n = y_train.shape[0]
y_0=0
for i in range(n):
    if y_train[i]==0:
        y_0+=1
y_1=n-y_0

def compute_phi(y_1,n):
    return y_1/n

def compute_mu_0(y_0,n):
    mu_0 = np.zeros((2, 1))
    for i in range(n):
        if(y_train[i]==0):
            mu_0 += x_train[i].reshape(2,1)
    return mu_0/y_0

def compute_mu_1(y_1,n):
    mu_1 = np.zeros((2, 1))
    for i in range(n):
        mu_1 += np.dot(x_train[i],y_train[i]).reshape(2,1)
    return mu_1/y_1

def compute_Sigma(n,mu_0,mu_1):
    cov = np.zeros((2, 2))    # convariance of two Gaussians
    for x, y in zip(x_train, y_train):
        x = x.reshape(2, 1)
        if y == 0:
            cov += np.matmul(x - mu_0, (x - mu_0).T)
        else:
            cov += np.matmul(x - mu_1, (x - mu_1).T)
    return cov/n

phi = compute_phi(y_1,n)
mu_0 = compute_mu_0(y_0,n)
mu_1 = compute_mu_1(y_1,n)
cov = compute_Sigma(n,mu_0,mu_1)
print(phi)
print(mu_0)
print(mu_1)
print(cov)

mean_0 = np.squeeze(mu_0)
Gaussian_0 = multivariate_normal(mean=mean_0, cov=cov)
mean_1 = np.squeeze(mu_1)
Gaussian_1 = multivariate_normal(mean=mean_1, cov=cov)


M = 100
X, Y = np.meshgrid(np.linspace(-10,-2,M), np.linspace(-2,8,M))
# 二维坐标数据
d = np.dstack([X,Y])
# 计算二维联合高斯概率
Z_0 = Gaussian_0.pdf(d).reshape(M,M)
Z_1 = Gaussian_1.pdf(d).reshape(M,M)

plt.figure()
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, edgecolors='white')
plt.contour(X, Y, Z_0,  alpha =1.0, zorder=10)
plt.contour(X, Y, Z_1,  alpha =1.0, zorder=10)
plt.show()