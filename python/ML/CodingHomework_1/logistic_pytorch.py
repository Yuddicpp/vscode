import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch 
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 使用pandas读取实验数据
df_opel_corsa_01 = pd.read_csv('Dataset_traffic-driving-style-road-surface-condition/opel_corsa_01.csv',delimiter=';')

df_opel_corsa_02 = pd.read_csv('Dataset_traffic-driving-style-road-surface-condition/opel_corsa_02.csv',delimiter=';')

df_peugeot_207_01 = pd.read_csv('Dataset_traffic-driving-style-road-surface-condition/peugeot_207_01.csv',delimiter=';')

df_peugeot_207_02 = pd.read_csv('Dataset_traffic-driving-style-road-surface-condition/peugeot_207_02.csv',delimiter=';')

# 实验数据处理，将结果划分为三维向量
def Road_surface_data_process(data_raw):
    y = pd.get_dummies(data_raw['roadSurface']).to_numpy()
    x = data_raw.iloc[:,:14].fillna(0).to_numpy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if type(x[i,j]) == str:
                x[i,j] = float(x[i,j].replace(',','.'))
    # print(x,y)
    return x,y

X,Y = Road_surface_data_process(pd.concat([df_opel_corsa_01, df_opel_corsa_02, df_peugeot_207_01, df_peugeot_207_02], axis=0))
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

#归一化
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit(x_train)
x_train_transformed = min_max_scaler.transform(x_train)
x_test_transformed = min_max_scaler.transform(x_test)

# GPU训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dl_train = DataLoader(TensorDataset(torch.tensor(x_train_transformed).float().to(device),torch.tensor(y_train).float().to(device)),shuffle = True, batch_size = 8)
dl_valid = DataLoader(TensorDataset(torch.tensor(x_test_transformed).float().to(device),torch.tensor(y_test).float().to(device)),shuffle = False, batch_size = 8)

# 创建网络
def create_net():
    net = nn.Sequential()
    net.add_module("linear1",nn.Linear(14,20))
    net.add_module("relu1",nn.ReLU())
    net.add_module("linear2",nn.Linear(20,30))
    net.add_module("relu2",nn.ReLU())
    net.add_module("linear3",nn.Linear(30,15))
    net.add_module("relu3",nn.ReLU())
    net.add_module("linear4",nn.Linear(15,6))
    net.add_module("relu4",nn.ReLU())
    net.add_module("linear5",nn.Linear(6,3))
    net.add_module("sigmoid",nn.Sigmoid())
    return net
    
net = create_net()

net = net.to(device) # 移动模型到cuda

# 训练模型

# net.load_state_dict(torch.load('logistic.pkl'))

# 损失和优化函数定义
loss_func  = torch.nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
acc = []

# 迭代次数
EPOCHS = 1000

#开始训练
for epoch in range(1,EPOCHS):  

    net.train()
    loss_sum = 0.0
    metric_sum = 0.0
    step = 1
    
    for step, (features,labels) in enumerate(dl_train, 1):
    
        # 梯度清零
        optimizer.zero_grad()

        # 正向传播求损失
        predictions = net(features)
        loss = loss_func(predictions,labels)
        metric = accuracy_score( np.rint(predictions.cpu().detach().numpy()),labels.cpu().detach().numpy())
        
        # 反向传播求梯度
        loss.backward()
        optimizer.step()

        # 打印batch级别日志
        loss_sum += loss.item()
        metric_sum += metric.item()


    net.eval()
    val_loss_sum = 0.0
    val_metric_sum = 0.0
    val_step = 1

    for val_step, (features,labels) in enumerate(dl_valid, 1):
        # 关闭梯度计算
        with torch.no_grad():
            predictions = net(features)
            val_loss = loss_func(predictions,labels)
            val_metric = accuracy_score( np.rint(predictions.cpu().detach().numpy()),labels.cpu().detach().numpy())
        val_loss_sum += val_loss.item()
        val_metric_sum += val_metric.item()
    
    # 打印epoch级别日志
    info = (epoch, loss_sum/step, metric_sum/step,val_loss_sum/val_step, val_metric_sum/val_step)
    print(("\nEPOCH = %d, loss = %.3f,"+ "Accuracy" + "  = %.3f, val_loss = %.3f, "+"val_"+ "Accuracy"+" = %.3f\n\n")%info) 
    acc.append(val_metric_sum/val_step)
    # print("-------------------------------------------------")

torch.save(net.state_dict(), "./logistic.pkl")

plt.plot(range(1,EPOCHS),acc)
plt.ylim(0, 1);
plt.xlabel("EPOCH")
plt.ylabel("Accuracy")
plt.show()

# Result:

#EPOCH = 999, loss = 0.026,Accuracy  = 0.983, val_loss = 0.053, val_Accuracy = 0.976
