from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from time import *
from scipy.sparse import data
from sklearn.metrics.pairwise import cosine_similarity



#work2
# 读取训练数据文件
df_train = pd.read_csv('E:\BaiduNetdiskWorkspace\研究生\课程\大数据分析(B)\HW2\Project2-data\data_train.csv',index_col=0)
# print(df_train)

# 读取测试数据文件
df_test = pd.read_csv('E:\BaiduNetdiskWorkspace\研究生\课程\大数据分析(B)\HW2\Project2-data\data_test.csv',index_col=0)
# print(df_test)

data_train = df_train.to_numpy()
sim = cosine_similarity(data_train)
# print(sim)

def score(i,j): #第i个用户的第j个电影
    Denominator = 0 #∑_k sim(X(i),X(k))score(k,j) 
    molecular = 0 #∑_k sim(X(i),X(k))
    users = ((np.argsort(sim[i,:]))[::-1])[1:500]#返回相似用户序列号
    for k in users:
        sim_ik = sim[i,k]
        Denominator+=sim_ik*data_train[k,j]
        molecular+=sim_ik
    return Denominator/molecular


# begin_time = time()
# print(score(0,6))
# end_time = time()
# print(df_train.loc[305344,'7'])
# print("Time:"+str(end_time-begin_time))
# 测试

# begin_time = time()
# RMSE = 0
# Test_num = 0
# data_test = df_test.to_numpy()
# data_test = data_test[0:1000,0:100]
# for i in range(data_test.shape[0]):
#     for j in range(data_test.shape[1]):
#         if(data_test[i,j]!=0):
#             RMSE+=(score(i,j)-data_test[i,j])**2
#             Test_num+=1
#             print(Test_num)

# end_time = time()
# print("Time:"+str(end_time-begin_time))
# print((RMSE/Test_num)**0.5)
# print(Test_num)


data_test = df_test.to_numpy()
score_all = np.zeros((10000,10000))
for i in range(10000):
    for j in range(10000):
        score_all[i,j] = score(i,j)

RMSE = np.sqrt(np.sum(np.sum(np.square(np.multiply(data_test > 0, score_all) - data_test)))/1719466)
print(RMSE)
