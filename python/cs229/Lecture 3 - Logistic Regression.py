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

lr = 0.01

x, y = make_blobs(n_samples=N_SAMPLES, centers=2, n_features=N_FEATURES, random_state=3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIEZ, random_state=0)
# plt.scatter(x[:,0], x[:, 1], c=y, edgecolors='white', marker='s')
# plt.show()

x_0_train = np.ones((x_train.shape[0],1), dtype=x_train.dtype)
x_0_test = np.ones((x_test.shape[0],1), dtype=x_test.dtype)
x_train = np.concatenate((x_0_train,x_train),axis=1)
x_test = np.concatenate((x_0_test,x_test),axis=1)

theta = np.random.rand(N_FEATURES+1).reshape(N_FEATURES+1,1)

def hypostheses(theta,x):
    k = np.matmul(x.T,theta)
    return 1/(1+exp(-k))

def Stochastic_update_weights(theta,h,y,x):
    k = lr*(h-y)*(x)
    return theta- k



for i in range(20):
    for x,y in zip(x_train,y_train):
        x = x.reshape(N_FEATURES+1,1)
        h = hypostheses(theta,x)
        Stochastic_update_weights(theta,h,y,x)

# h_test = np.zeros(y_test.shape[0], dtype=y_test.dtype)
h_test = y_test
i = 0
for x,y in zip(x_test,y_test):
    x = x.reshape(N_FEATURES+1,1)
    h = hypostheses(theta,x)
    h_test[i] = round(h)
    i+=1

plt.figure(0)
plt.scatter(x_test[:, 1], x_test[:, 2], c=h_test, edgecolors='white', marker='s')
plt.figure(1)
plt.scatter(x_test[:, 1], x_test[:, 2], c=y_test, edgecolors='white', marker='s')

plt.show()