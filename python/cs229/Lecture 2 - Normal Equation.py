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