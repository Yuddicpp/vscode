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
    for i in range(n):
        x_train[i].reshape(2,1)
        if(y_train[i]==0):
            cov += np.matmul((x_train[i]-mu_0),(x_train[i]-mu_0).T)
        elif(y_train[i]==1):
            cov += np.matmul((x_train[i]-mu_1),(x_train[i]-mu_1).T)
    return cov/n

mu_0 = compute_mu_0(y_0,n)
mu_1 = compute_mu_1(y_1,n)
cov = compute_Sigma(n,mu_0,mu_1)
print(mu_0,mu_1,cov)



def tow_d_gaussian(x, mu, COV):

    n = mu.shape[0]
    COV_det = np.linalg.det(COV)
    COV_inv = np.linalg.inv(COV)
    N = np.sqrt((2*np.pi)**n*COV_det)

    fac = np.einsum('...k,kl,...l->...',x-mu,COV_inv,x-mu)

    return np.exp(-fac/2)/N

if __name__ == '__main__':

    # obtain rule of paramters exploiting log-likelihood
    # fi, miu_0, miu_1, COV = update_parameters(X_train, Y_train, fi, miu_0, miu_1, COV)

    # plotting
    fig =plt.figure()
    # ax = fig.gca(projection='3d') # 3d plotting
    ax = fig.gca()
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolors='white')

    N = 60
    X = np.linspace(-10,-2,N)
    Y = np.linspace(-2,8,N)
    X,Y = np.meshgrid(X,Y)

    pos = np.empty(X.shape+(2,))
    pos[:,:,0]= X
    pos[:,:,1] = Y

    miu_0 = np.reshape(mu_0, (1, 2))[0]
    miu_1 = np.reshape(mu_1, (1, 2))[0]

    Z1 = tow_d_gaussian(pos, miu_0, cov)
    Z2 = tow_d_gaussian(pos, miu_1, cov)
    
    cset = ax.contour(X,Y,Z1,zdir='z',offset=-0.15)
    cset = ax.contour(X,Y,Z2,zdir='z',offset=-0.15)

   
    # 3d plotting
    # ax.plot_surface(X,Y,Z1,rstride=3,cstride=3,linewidth=1,antialiased =True)
    # ax.plot_surface(X,Y,Z2,rstride=3,cstride=3,linewidth=1,antialiased =True)
    
    # ax.set_zlim(-0.15,0.2)
    # ax.set_zticks(np.linspace(0,0.2,5))
    # ax.view_init(12,-12)


    plt.show()