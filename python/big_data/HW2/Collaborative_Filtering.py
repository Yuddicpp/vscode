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
    users = ((np.argsort(sim[i,:]))[::-1])#返回相似用户序列号  
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

begin_time = time()
RMSE = 0
Test_num = 0
data_test = df_test.to_numpy()
for i in range(data_test.shape[0]):
    for j in range(data_test.shape[1]):
        if(data_test[i,j]!=0):
            RMSE+=(score(i,j)-data_test[i,j])**2
            Test_num+=1

end_time = time()
print("Time:"+str(end_time-begin_time))
print((RMSE/Test_num)**0.5)
print(Test_num)

#50 
# Time:1253.5741956233978
# 1.972846441587203
# 1719466

#30
# Time:1268.786537885666
# 1.9664577108322285
# 1719466

#10
# Time:1307.0318825244904
# 1.9917049156387596
# 1719466

#100
# Time:1413.5078313350677
# 1.990252557854739
# 1719466