from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from time import *
from sklearn.metrics.pairwise import cosine_similarity
# df_train = pd.read_csv('Project2-data/netflix_train.txt',sep = ' ',names=['users','movie_title','rank','data'])
# df_users = pd.read_csv('Project2-data/users.txt',names = ['users'])

# data_train = pd.DataFrame(0,index = np.array(df_users['users']),columns=np.array(range(1,10001)))


# for index, row in df_train.iterrows():
#     print(index,row['rank'])
#     data_train.loc[row['users'],row['movie_title']] = row['rank']

# print(data_train)
# data_train.to_csv("data_train.csv")

# 读取训练数据文件
df_train = pd.read_csv('~/work/data_train.csv',index_col=0)
# print(df_train)

# 读取测试数据文件
df_test = pd.read_csv('data_test.csv',index_col=0)
# print(df_test)

df_users = pd.DataFrame(range(1,10001),index =df_train.index,columns=['order'])

data_train = df_train.to_numpy()
sim = cosine_similarity(data_train)

def sim(x,y):
    return (np.dot(x,y))/((np.linalg.norm(x))*(np.linalg.norm(y)))


def score(i,j):
    num = df_users.loc[i,'order']-1 #第几个用户
    Denominator = 0 #∑_k sim(X(i),X(k))score(k,j) 
    molecular = 0 #∑_k sim(X(i),X(k))
    users = ((np.argsort(sim[num,:]))[::-1])[:10]#返回用户序列号
    for m in range(10):
        k = users[m]
        sim_ik = sim[num,k]
        Denominator+=sim_ik*data_train[k,j-1]
        molecular+=sim_ik
    # sim_i = pd.DataFrame(0,index = np.array(df_train.index),columns=['sim'])
    # # print(sim_i)
    # for index, row in sim_i.iterrows():
    #     sim_i.loc[index,'sim'] = sim((df_train.loc[i,:]).to_numpy(),df_train.loc[index,:].to_numpy())
    #     if(sim_i.loc[index,'sim']<1 and sim_i.loc[index,'sim']>0.5):
    #         Denominator+=(row['sim']*df_train.loc[index,str(j)])
    #         molecular+=row['sim']
    return Denominator/molecular


begin_time = time()
print(score(305344,1))
end_time = time()
print(df_train.loc[305344,'1'])
print("Time:"+str(end_time-begin_time))
# 测试
# RMSE = 0;
# Test_num = 0;
# for index, row in df_test.iterrows():
#     for i in range(1,10001):
#         if(row[str(i)]!=0):
#             RMSE+=(score(index,i)-row[str(i)])**2
#             Test_num+=1
#             print(Test_num,RMSE)

# print((RMSE/Test_num)**0.5)
# print(Test_num)

##




