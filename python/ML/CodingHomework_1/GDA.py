import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 使用pandas读取实验数据
df_opel_corsa_01 = pd.read_csv('Dataset_traffic-driving-style-road-surface-condition/opel_corsa_01.csv',delimiter=';')

df_opel_corsa_02 = pd.read_csv('Dataset_traffic-driving-style-road-surface-condition/opel_corsa_02.csv',delimiter=';')

df_peugeot_207_01 = pd.read_csv('Dataset_traffic-driving-style-road-surface-condition/peugeot_207_01.csv',delimiter=';')

df_peugeot_207_02 = pd.read_csv('Dataset_traffic-driving-style-road-surface-condition/peugeot_207_02.csv',delimiter=';')

# 实验数据处理，将‘AggressiveStyle’设为1，另一状态设为0
def Road_surface_data_process(data_raw):
    roadSurface_to_state = {
        'EvenPaceStyle' : 0,
        'AggressiveStyle' : 1
    }
    y = data_raw['drivingStyle'].map(roadSurface_to_state).fillna(0).to_numpy()
    x = data_raw.iloc[:,:14].fillna(0).to_numpy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if type(x[i,j]) == str:
                x[i,j] = float(x[i,j].replace(',','.'))  #由于实验数据中小数点为‘,’,更改为‘.’
    return x,y


X,Y = Road_surface_data_process(pd.concat([df_opel_corsa_01, df_opel_corsa_02, df_peugeot_207_01, df_peugeot_207_02], axis=0))
# 分为数据集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


# 计算高斯模型的参数
def train(x_train,y_train):

    n = y_train.shape[0]
    y_0=0
    for i in range(n):
        if y_train[i]==0:
            y_0+=1
    y_1=n-y_0
    phi = y_1/n

    mu_0 = np.zeros((14, 1))
    for i in range(n):
        if(y_train[i]==0):
            mu_0 = mu_0 + np.expand_dims(x_train[i],axis=1)
    mu_0 = mu_0/y_0

    mu_1 = np.zeros((14, 1))
    for i in range(n):
        if(y_train[i]==1):
            mu_1 = mu_1 + np.expand_dims(x_train[i],axis=1)
    mu_1 = mu_1/y_1

    cov = np.zeros((14, 14))   # convariance of two Gaussians
    for x, y in zip(x_train, y_train):
        x = np.expand_dims(x,1)
        if y == 0:
            cov = cov + np.matmul(x - mu_0, (x - mu_0).T)
        else:
            cov = cov + np.matmul(x - mu_1, (x - mu_1).T)
    cov =  cov/n

    return phi,mu_0.astype('float32'),mu_1.astype('float32'),cov.astype('float32')

#计算高斯模型下的概率值
def compute_p(x,mu_0,mu_1,cov,phi):
    p0 = np.exp((-0.5*np.dot(np.dot((np.transpose(x-mu_0)),cov),(x-mu_0)))[0,0])*(1-phi)
    p1 = np.exp((-0.5*np.dot(np.dot((np.transpose(x-mu_1)),cov),(x-mu_1)))[0,0])*phi
    if(p0>p1):
        return 0
    else:
        return 1


#测试函数
def predict(x_test,y_test):
    phi,mu_0,mu_1,cov = train(x_train,y_train)
    y_pre = []
    for i in range(x_test.shape[0]):
        y_pre.append(compute_p(np.expand_dims(x_test[i],1),mu_0,mu_1,np.matrix(cov).I,phi))
    
    print(accuracy_score(y_pre,y_test))


predict(x_test,y_test)

