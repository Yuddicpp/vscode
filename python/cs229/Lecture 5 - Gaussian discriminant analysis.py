# /*
#  * @Author: Yuddi 
#  * @Date: 2021-09-03 16:00:54 
#  * @Last Modified by:    
#  * @Last Modified time: 2021-09-03 16:00:54 
#  */



from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.stats import multivariate_normal


np.random.seed(1)


phi = 0                                             # probablity of y = 1
mu_0 = np.zeros((2, 1))  # mean of y = 0
mu_1 = np.zeros((2, 1))  # mean of y = 1
COV = np.zeros((2, 2))    # convariance of two Gaussians


x, y = make_blobs(n_samples=300, n_features=2, centers=2, random_state=12)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolors='white')
plt.show()
