from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from time import *
from scipy.sparse import data
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt




#work2
# 读取训练数据文件
# df_train = pd.read_csv('~/work/data_train.csv',index_col=0)
# print(df_train)

# 读取测试数据文件
df_test = pd.read_csv('~/work/data_test.csv',index_col=0)
# print(df_test)

# data_train = df_train.to_numpy()
data_test = df_test.to_numpy()



begin_time = time()
RMSE = 0
Test_num = 0
data_test = df_test.to_numpy()
for i in range(data_test.shape[0]):
    for j in range(data_test.shape[1]):
        if(data_test[i,j]!=0):
            RMSE+=(3-data_test[i,j])**2
            Test_num+=1

end_time = time()
print("Time:"+str(end_time-begin_time))
print((RMSE/Test_num)**0.5)
print(Test_num)