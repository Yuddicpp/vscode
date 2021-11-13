from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from time import *
from scipy.sparse import data
from sklearn.metrics.pairwise import cosine_similarity



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

def score(i,j): #第i个用户的第j个电影
    Denominator = 0 #∑_k sim(X(i),X(k))score(k,j) 
    molecular = 0 #∑_k sim(X(i),X(k))
    users = ((np.argsort(sim[i,:]))[::-1])[1:50]#返回相似用户序列号
    for k in users:
        sim_ik = sim[i,k]
        Denominator+=sim_ik*data_train[k,j]
        molecular+=sim_ik
    # sim_i = pd.DataFrame(0,index = np.array(df_train.index),columns=['sim'])
    # # print(sim_i)
    # for index, row in sim_i.iterrows():
    #     sim_i.loc[index,'sim'] = sim((df_train.loc[i,:]).to_numpy(),df_train.loc[index,:].to_numpy())
    #     if(sim_i.loc[index,'sim']<1 and sim_i.loc[index,'sim']>0.5):
    #         Denominator+=(row['sim']*df_train.loc[index,str(j)])
    #         molecular+=row['sim']
    return Denominator/molecular


# begin_time = time()
# print(score(0,6))
# end_time = time()
# print(df_train.loc[305344,'7'])
# print("Time:"+str(end_time-begin_time))
# 测试


RMSE = 0
Test_num = 0
data_test = df_test.to_numpy()
for i in range(10000):
    for j in range(10000):
        if(data_test[i,j]!=0):
            RMSE+=(score(i,j)-data_test[i,j])**2
            Test_num+=1
            print(Test_num)

print((RMSE/Test_num)**0.5)
print(Test_num)


