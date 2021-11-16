from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from time import *
from scipy.sparse import data
from sklearn.metrics.pairwise import cosine_similarity



#work2
# 读取100个用户
# 读取训练数据文件
df_train = pd.read_csv('~/work/data_train.csv',index_col=0) #100*10000
# print(df_train)

# 读取测试数据文件
df_test = pd.read_csv('~/work/data_test.csv',index_col=0) #100*10000
# print(df_test)

data_train = df_train.to_numpy()
sim = cosine_similarity(data_train) #100*100

def score(i,j): #第i个用户的第j个电影
    Denominator = 0 #∑_k sim(X(i),X(k))score(k,j) 
    molecular = 0 #∑_k sim(X(i),X(k))
    users = ((np.argsort(sim[i,:]))[::-1])[1:20]#返回相似用户序列号 #10 1.99 #5 2.072 #100 1.99 #20
    for k in users:
        sim_ik = sim[i,k]
        Denominator+=sim_ik*data_train[k,j]
        molecular+=sim_ik
    return Denominator/molecular+np.mean(data_train[i,:])


# begin_time = time()
# print(score(0,6))
# end_time = time()
# print(df_train.loc[305344,'7'])
# print("Time:"+str(end_time-begin_time))
# 测试



RMSE = 0
Test_num = 0
data_test = df_test.to_numpy()
begin_time = time()
for i in range(10000):
    for j in range(10000):
        if(data_test[i,j]!=0):
            k = (score(i,j)-data_test[i,j])**2
            # print(k)
            RMSE+= k
            Test_num+=1

print((RMSE/Test_num)**0.5)
print(Test_num)
end_time = time()
print("Time:"+str(end_time-begin_time))


