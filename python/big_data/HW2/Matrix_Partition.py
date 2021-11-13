from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from time import *
from scipy.sparse import data
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt



#work2
# 读取训练数据文件
df_train = pd.read_csv('E:\BaiduNetdiskWorkspace\研究生\课程\大数据分析(B)\HW2\Project2-data\data_train.csv',index_col=0)
# print(df_train)

# 读取测试数据文件
df_test = pd.read_csv('E:\BaiduNetdiskWorkspace\研究生\课程\大数据分析(B)\HW2\Project2-data\data_test.csv',index_col=0)
# print(df_test)

data_train = df_train.to_numpy()
data_test = df_test.to_numpy()

alpha = 0.0001
lamda = 1
K = 100
EPOCH = 50



A = data_train > 0
U = np.random.randn(10000, K)*0.1
V = np.random.randn(10000, K)*0.1

J = np.zeros(EPOCH)
RMSE = np.zeros(EPOCH)
for i in range(EPOCH):
    dU = np.dot(np.multiply(A, (np.dot(U, V.T) - data_train)), V) + 2 * lamda * U
    dV = np.dot(np.multiply(A, (np.dot(U, V.T) - data_train)), U) + 2 * lamda * V
    U = U - alpha * dU # Learning rate decay
    V = V - alpha * dV
    J[i] = 1/2*np.sum(np.sum(np.square(np.multiply(A, (data_train - np.dot(U, V.T)))))) + lamda * np.sum(np.sum(np.square(U)))\
           + lamda * np.sum(np.sum(np.square(V)))
    RMSE[i] = np.sqrt(np.sum(np.sum(np.square(np.multiply(data_test > 0, np.dot(U, V.T)) - data_test)))/1719466)
    print(i)
# Visualization
X = np.dot(U, V.T)
plt.plot(range(EPOCH), RMSE)
plt.show()
plt.plot(range(EPOCH), J)
plt.show()
print(RMSE[EPOCH-1])