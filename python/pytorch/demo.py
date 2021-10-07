import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import copy
import torch 
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset,sampler
from sklearn.metrics import accuracy_score

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def question_3A(data_raw):
    # 正态分布检测

    # 画概率密度函数
    data = data_raw.iloc[:,6]
    min = np.min(data.to_numpy())
    max = np.max(data.to_numpy())
    density = 15
    k = (max-min)/density
    x = []
    y = []
    # print(data_raw[(data_raw['平均年龄']>=(min+k*3))&(data_raw['平均年龄']<=(min+k*4))].shape[0])
    for i in range(0,density-1):
        x.append(min+i*k)
        y.append(data_raw[(data_raw['平均年龄']>=(min+k*i))&(data_raw['平均年龄']<=(min+(i+1)*k))].shape[0])
    
    plt.ion()
    plt.bar(x,y)
    plt.pause(1)
    plt.close()

    # python自带正态分布检测，p>0.05则是满足正态分布
    print(stats.kstest(data,'norm',(data_raw.iloc[:,6].mean(),data_raw.iloc[:,6].std())))
    print(stats.normaltest(data))
    print(stats.shapiro(data))

def question_3B(data):
    for i in range(1,6):
        print(i)
        plt.xlabel('Average Age')
        plt.ylabel('PDF_categoty='+str(i))
        question_3A(data[data['群类别']==i])
    
    stat, p = stats.levene(data[data['群类别']==1].iloc[:,6],data[data['群类别']==2].iloc[:,6],data[data['群类别']==3].iloc[:,6],data[data['群类别']==4].iloc[:,6],data[data['群类别']==5].iloc[:,6])
    print(p)


def question_3C(data_raw):
    # 读取群类别和平均年龄到data
    k = 5
    data = pd.concat([data_raw.iloc[:,1],data_raw.iloc[:,6]],axis=1)
    # 求整个data的平均年龄的平均值
    data_mean = data.iloc[:,1].mean()
    
    # 整个
    SST = 0
    for i in range(data.iloc[:,1].shape[0]):
        SST+=(data.iloc[i,1]-data_mean)**2
    
    # 组间
    df_B = k - 1
    SSB = 0
    for i in range(1,6):
        temp = data[data.iloc[:,0]==i].iloc[:,1]
        SSB+=((temp.mean()-data_mean)**2)*(temp.shape[0])

    # 组内
    df_W = data.iloc[:,1].shape[0] - k
    # SSW = SST - SSB
    SSW = 0
    for i in range(data.iloc[:,1].shape[0]):
        SSW+=(data.iloc[i,1]-data[data.iloc[:,0]==data.iloc[i,0]].iloc[:,1].mean())**2
    

    MS_B = SSB/df_B
    MS_W = SSW/df_W
    
    F = MS_B/MS_W
    # print(F)

    f, p = stats.f_oneway(data[data.iloc[:,0]==1].iloc[:,1],data[data.iloc[:,0]==2].iloc[:,1],data[data.iloc[:,0]==3].iloc[:,1],data[data.iloc[:,0]==4].iloc[:,1],data[data.iloc[:,0]==5].iloc[:,1])
    # print(f,p)

    result = {
        'SS':{'Between': SSB, 'Within': SSW, 'Total': SST},
        'df':{'Between': df_B, 'Within': df_W, 'Total': df_W+df_B},
        'MS':{'Between': MS_B, 'Within': MS_W},
        'F' :{'Between': f},
        'P' :{'Between': p}
    }
    result = pd.DataFrame(result)
    return result

def create_net_4A():
    net = nn.Sequential()
    net.add_module("linear1",nn.Linear(1,3))
    return net

def question_4A(data_raw):
    data = data_raw[data_raw['会话数']>=20]
    x = data.iloc[:,6].to_numpy().reshape(1247,1)
    labels = data.iloc[:,11:14].to_numpy()
    train = DataLoader(TensorDataset(torch.tensor(x).float(),torch.tensor(labels).float()),shuffle=True, batch_size=8)
    net = create_net_4A()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(params=net.parameters(),lr = 1e-3)
    net.train()
    for i in range(50):
        loss_array = np.ones([1,3])
        loss_sum = 0

        for step, (features,labels) in enumerate(train, 1):
            optimizer.zero_grad()

            predictions = net(features)
            loss = loss_func(predictions,labels)
            loss_array = np.vstack((loss_array,(predictions-labels).detach().numpy()))

            loss.backward()
            optimizer.step()

            loss_sum+=loss.item()
        loss_array = loss_array[1:,:]
        print("Loss:"+str(loss_sum/step))
        print(loss_array)
        print("Variance of errors(Col12):"+str(np.var(loss_array[:,0])))
        print("Variance of errors(Col13):"+str(np.var(loss_array[:,1])))
        print("Variance of errors(Col14):"+str(np.var(loss_array[:,2])))
    
    print(net.state_dict())

def create_net_4B():
    net = nn.Sequential()
    net.add_module("linear1",nn.Linear(8,10))
    net.add_module("relu2",nn.ReLU())
    net.add_module("linear3",nn.Linear(10,3))
    return net

def question_4B(data):
    # data = data_raw[data_raw['会话数']>=20]
    x = data.iloc[:,2:10].to_numpy()
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    data.iloc[:,10].apply(max_min_scaler)
    for i in range(x.shape[0]):
        x[i,:] = x[i,:]*(data.iloc[:,10].to_numpy()[i])
    labels = data.iloc[:,11:14].to_numpy()
    train = DataLoader(TensorDataset(torch.tensor(x).float(),torch.tensor(labels).float()),shuffle=True, batch_size=8)
    net = create_net_4B()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(params=net.parameters(),lr = 1e-3)
    net.train()
    for i in range(50):
        loss_array = np.ones([1,3])
        loss_sum = 0

        for step, (features,labels) in enumerate(train, 1):
            optimizer.zero_grad()

            predictions = net(features)
            loss = loss_func(predictions,labels)
            loss_array = np.vstack((loss_array,(predictions-labels).detach().numpy()))

            loss.backward()
            optimizer.step()

            loss_sum+=loss.item()
        loss_array = loss_array[1:,:]
        print("Loss:"+str(loss_sum/step))
        print("Variance of errors(Col12):"+str(np.var(loss_array[:,0])))
        print("Variance of errors(Col13):"+str(np.var(loss_array[:,1])))
        print("Variance of errors(Col14):"+str(np.var(loss_array[:,2])))
    
    # print(net.state_dict())

def create_net_4C():
    net = nn.Sequential()
    net.add_module("linear1",nn.Linear(12,50))
    net.add_module("relu1",nn.ReLU())
    net.add_module("linear2",nn.Linear(50,10))
    net.add_module("relu2",nn.ReLU())
    net.add_module("linear3",nn.Linear(10,1))
    return net

def question_4C(data_raw):
    data = data_raw[data_raw['群类别']<=4]
    data = data.sample(frac=1)
    x = data.iloc[:,2:14].to_numpy()
    labels = data[['群类别']].to_numpy()

    split_num = int(x.shape[0] * 0.9)
    index_list = list(range(x.shape[0]))
    train_idx, valid_idx = index_list[:split_num], index_list[split_num:]

    tr_sampler = sampler.SubsetRandomSampler(train_idx)
    val_sampler = sampler.SubsetRandomSampler(valid_idx)

    train = DataLoader(TensorDataset(torch.tensor(x).float(),torch.tensor(labels).float()), batch_size=8,sampler=tr_sampler)
    test = DataLoader(TensorDataset(torch.tensor(x).float(),torch.tensor(labels).float()), batch_size=8,sampler=val_sampler)
    
    net = create_net_4C()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(params=net.parameters(),lr = 1e-3)
    net.load_state_dict(torch.load('question_4C.pkl'))
    net.train()
    for i in range(10000):
        loss_sum = 0

        for step, (features,labels) in enumerate(train, 1):
            optimizer.zero_grad()

            predictions = net(features)
            loss = loss_func(predictions,labels)

            loss.backward()
            optimizer.step()

            loss_sum+=loss.item()
        print("Epoch:"+str(i)+"  Loss:"+str(loss_sum/step))
    
    # print(net.state_dict())

    net.eval()
    loss_sum = 0
    metric_sum = 0
    metric_func = lambda y_pred,y_true: accuracy_score(y_true.data.numpy(),y_pred.data.numpy())
    for step, (features,labels) in enumerate(test, 1):
        with torch.no_grad():

            predictions = net(features).round()
            # print(predictions,labels)
            loss = loss_func(predictions,labels)
            
            metric = metric_func(predictions,labels)
            # loss.backward()
            # optimizer.step()

            loss_sum+=loss.item()
            metric_sum+=metric.item()
    print("Loss:"+str(loss_sum/step)+"   acu:"+str(metric_sum/step))

    torch.save(net.state_dict(), "./question_4C.pkl")

def question_5(data):
    data_sample = data.sample(frac=0.1)
    print(question_3C(data_sample))


data = pd.read_excel('data.xlsx')
# 不满足正态分布
# question_3A(data)
# 2,3满足正态分布
# question_3B(data)
# print(question_3C(data))
# question_4A(data)
# question_4B(data)
# question_4C(data)
question_5(data)