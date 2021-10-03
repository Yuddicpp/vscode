import os, sys
from pandas.io.parsers import read_csv

from torch._C import wait
os.chdir(sys.path[0])

import datetime
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import torch 
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset

from sklearn.metrics import accuracy_score

# Survived:0代表死亡，1代表存活【y标签】
# Pclass:乘客所持票类，有三种值(1,2,3) 【转换成onehot编码】
# Name:乘客姓名 【舍去】
# Sex:乘客性别 【转换成bool特征】
# Age:乘客年龄(有缺失) 【数值特征，添加“年龄是否缺失”作为辅助特征】
# SibSp:乘客兄弟姐妹/配偶的个数(整数值) 【数值特征】
# Parch:乘客父母/孩子的个数(整数值)【数值特征】
# Ticket:票号(字符串)【舍去】
# Fare:乘客所持票的价格(浮点数，0-500不等) 【数值特征】
# Cabin:乘客所在船舱(有缺失) 【添加“所在船舱是否缺失”作为辅助特征】
# Embarked:乘客登船港口:S、C、Q(有缺失)【转换成onehot编码，四维度 S,C,Q,nan】


# dftrain_raw = pd.read_csv('./data/titanic/train.csv')
# dftest_raw = pd.read_csv('./data/titanic/test.csv')

# # Process data
# def preprocessing(dfdata):
#     dfresult= pd.DataFrame()

#     #Pclass
#     # get_dummies():Convert categorical variable into dummy/indicator variables.
#     dfPclass = pd.get_dummies(dfdata['Pclass'])
#     dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ]
#     # concat():Concatenate pandas objects along a particular axis with optional set logic along the other axes.
#     dfresult = pd.concat([dfresult,dfPclass],axis = 1)

#     #Sex
#     dfSex = pd.get_dummies(dfdata['Sex'])
#     dfresult = pd.concat([dfresult,dfSex],axis = 1)

#     #Age
#     # fillna():Fill NA/NaN values using the specified method.
#     dfresult['Age'] = dfdata['Age'].fillna(0)
#     dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

#     #SibSp,Parch,Fare
#     dfresult['SibSp'] = dfdata['SibSp']
#     dfresult['Parch'] = dfdata['Parch']
#     dfresult['Fare'] = dfdata['Fare']

#     #Carbin
#     dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')

#     #Embarked
#     dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True)
#     dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
#     dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)

#     return dfresult


# x_train = preprocessing(dftrain_raw).values
# y_train = dftrain_raw[['Survived']].values

# x_test = preprocessing(dftest_raw).values
# y_test = dftest_raw[['Survived']].values

# print("x_train.shape =", x_train.shape )
# print("x_test.shape =", x_test.shape )

# print("y_train.shape =", y_train.shape )
# print("y_test.shape =", y_test.shape )

# dl_train = DataLoader(TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).float()),shuffle = True, batch_size = 8)
# dl_valid = DataLoader(TensorDataset(torch.tensor(x_test).float(),torch.tensor(y_test).float()),shuffle = False, batch_size = 8)

# def create_net():
#     net = nn.Sequential()
#     net.add_module("linear1",nn.Linear(15,20))
#     net.add_module("relu1",nn.ReLU())
#     net.add_module("linear2",nn.Linear(20,15))
#     net.add_module("relu2",nn.ReLU())
#     net.add_module("linear3",nn.Linear(15,1))
#     net.add_module("sigmoid",nn.Sigmoid())
#     return net
    
# net = create_net()

# # 网络设置
# # 损失函数
# loss_func = nn.BCELoss()
# # 优化器
# optimizer = torch.optim.Adam(params=net.parameters(),lr = 0.01)
# # 精度
# metric_func = lambda y_pred,y_true: accuracy_score(y_true.data.numpy(),y_pred.data.numpy()>0.5)
# metric_name = "accuracy"

# epochs = 10
# step = 1

# # Train
# # The enumerate object yields pairs containing a count (from start, which defaults to zero) and a value yielded by the iterable argument.
# # step_num = data_num / batch_size

# for epoch in range(1,epochs+1):
#     loss_sum = 0
#     metric_sum = 0
#     net.train()
#     for step, (features,labels) in enumerate(dl_train, 1):

#         # 梯度清零
#         optimizer.zero_grad()

#         # 正向传播求损失
#         predictions = net(features)
#         loss = loss_func(predictions,labels)
#         metric = metric_func(predictions,labels)
        
#         # 反向传播求梯度
#         loss.backward()
#         optimizer.step()

#         loss_sum += loss.item()
#         metric_sum += metric.item()

#     # print(("Train:[Epoch = %d] loss: %.3f, "+metric_name+": %.3f") %(epoch, loss_sum/step, metric_sum/step))


#     # Validation
#     # with torch.zero_grad()主要是用于停止autograd模块的工作，以起到加速和节省显存的作用，具体行为就是停止gradient计算，从而节省了GPU算力和显存，但是并不会影响dropout和batchnorm层的行为。
#     val_loss_sum = 0
#     val_metric_sum = 0
#     net.eval()
#     for step, (features,labels) in enumerate(dl_valid, 1):
#         with torch.no_grad():
#             predictions = net(features)
#             loss = loss_func(predictions,labels)
#             metric = metric_func(predictions,labels)

#         val_loss_sum += loss.item()
#         val_metric_sum += metric.item()
#     print(("Validation:[Epoch = %d] loss: %.3f, "+metric_name+": %.3f") %(epoch, val_loss_sum/step, val_metric_sum/step))

# # save model
# torch.save(net.state_dict(), "./data/net_parameter.pkl")





# 自己尝试编写
def data_process(input_data):    
    output_data = pd.DataFrame()

    # Pclass
    Pclass_data = pd.get_dummies(input_data['Pclass'])
    Pclass_data.columns = ['Pclass_'+str(x) for x in Pclass_data.columns]
    output_data = pd.concat([output_data,Pclass_data],axis=1)

    # Sex
    Sex_data = pd.get_dummies(input_data['Sex'])
    output_data = pd.concat([output_data,Sex_data],axis=1)

    # Age
    output_data = pd.concat([output_data,input_data['Age'].fillna(0)],axis=1)
    output_data['Age_null'] = pd.isna(input_data['Age']).astype('int32')

    # SibSp,Parch,Fare
    output_data = pd.concat([output_data,input_data['SibSp'],input_data['Parch'],input_data['Fare']],axis=1)

    # Cabin
    output_data['Cabin_null'] = pd.isna(input_data['Cabin']).astype('int32')

    # Embarked
    Embarked_data = pd.get_dummies(input_data['Embarked'])
    Embarked_data.columns = ['Embarked_' + str(x) for x in Embarked_data.columns]
    output_data = pd.concat([output_data,Embarked_data],axis=1)
    output_data['Embarked_null'] = pd.isna(input_data['Embarked']).astype('int32')

    return output_data


def create_net():
    net = nn.Sequential()
    net.add_module("linear1",nn.Linear(15,50))
    net.add_module("relu1",nn.ReLU())
    net.add_module("linear2",nn.Linear(50,10))
    net.add_module("relu2",nn.ReLU())
    net.add_module("linear3",nn.Linear(10,1))
    net.add_module("sigmoid",nn.Sigmoid())
    return net


train_data = pd.read_csv('./data/titanic/train.csv')
test_data = pd.read_csv('./data/titanic/test.csv')

x_train = data_process(train_data).values
y_train = train_data[['Survived']].values

x_test = data_process(test_data).values
y_test = test_data[['Survived']].values

print(x_train.shape,y_train.shape)
train = DataLoader(TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).float()),shuffle=True, batch_size=8)
test = DataLoader(TensorDataset(torch.tensor(x_test).float(),torch.tensor(y_test).float()),shuffle=True, batch_size=8)

net = create_net()

loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(params=net.parameters(),lr = 0.001)
metric_func = lambda y_pred,y_true: accuracy_score(y_true.data.numpy(),y_pred.data.numpy()>0.5)

epoches = 100

for i in range(1,epoches+1):
    loss_sum = 0
    metric = 0
    net.train()
    for step, (features,labels) in enumerate(train, 1):
        optimizer.zero_grad()

        predictions = net(features)
        loss = loss_func(predictions,labels)
        metric += metric_func(predictions,labels)

        loss.backward()
        optimizer.step()

        loss_sum+=loss.item()
    
    print(("Epoch:%d Loss: %.3f Metric: %.3f")%(i,loss_sum/step,metric/step))
    loss_sum = 0
    metric = 0
        

loss_sum = 0
metric = 0
net.eval()
for step, (features,labels) in enumerate(test, 1):
    optimizer.zero_grad()

    predictions = net(features)
    loss = loss_func(predictions,labels)
    metric += metric_func(predictions,labels)

    loss.backward()
    optimizer.step()

    loss_sum+=loss.item()

print(("Test Loss: %.3f Metric: %.3f")%(loss_sum/step,metric/step))
