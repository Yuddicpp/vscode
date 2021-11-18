from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from time import *
from scipy.sparse import data
from sklearn.metrics.pairwise import cosine_similarity
import random



#work2
# 读取训练数据文件
df_train = pd.read_csv('~/work/data_train.csv',index_col=0)
# print(df_train)

# 读取测试数据文件
df_test = pd.read_csv('~/work/data_test.csv',index_col=0)
# print(df_test)

data_train = df_train.to_numpy()
sim = cosine_similarity(data_train)
# print(sim)

F = 20#取相似度前几名
def score(i,j): #第i个用户的第j个电影
    Denominator = 0 #∑_k sim(X(i),X(k))score(k,j) 
    molecular = 0 #∑_k sim(X(i),X(k))
    users = (np.argsort(sim[i,:])[::-1])#返回相似用户序列号 
    num = 0
    m = 0
    while(num<F and m<data_train.shape[0]-1):
        k = users[m+1]
        m+=1
        if(data_train[k,j]!=0):
            Denominator+=(sim[i,k]*data_train[k,j])
            molecular+=sim[i,k]
            num+=1
    if(molecular!=0):
        return Denominator/molecular
    else:
        return (np.mean(data_train[:,j])+np.mean(data_train[i:,]))/2


# begin_time = time()
# print(score(0,6))
# end_time = time()
# print(df_train.loc[305344,'7'])
# print("Time:"+str(end_time-begin_time))
# 测试

begin_time = time()
RMSE = 0
Test_num = 0
data_test = df_test.to_numpy()
for i in range(data_test.shape[0]):
    for j in range(data_test.shape[1]):
        if(data_test[i,j]!=0):
            RMSE+=(score(i,j)-data_test[i,j])**2
            Test_num+=1
            if(Test_num%10000==0):
                print(Test_num/10000)

end_time = time()
print("Time:"+str(end_time-begin_time))
print((RMSE/Test_num)**0.5)
print(Test_num)


#Baseline 全取3
# RMSE:1.1715111196739934

#Baseline 随机数
# RMSE:1.8373338676560507
