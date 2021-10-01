import os, sys
os.chdir(sys.path[0])

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


dftrain_raw = pd.read_csv('./data/titanic/train.csv')
dftest_raw = pd.read_csv('./data/titanic/test.csv')

# Process data
def preprocessing(dfdata):
    dfresult= pd.DataFrame()

    #Pclass
    # get_dummies():Convert categorical variable into dummy/indicator variables.
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ]
    # concat():Concatenate pandas objects along a particular axis with optional set logic along the other axes.
    dfresult = pd.concat([dfresult,dfPclass],axis = 1)

    #Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult,dfSex],axis = 1)

    #Age
    # fillna():Fill NA/NaN values using the specified method.
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    #SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    #Carbin
    dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)

    return dfresult


x_train = preprocessing(dftrain_raw).values
y_train = dftrain_raw[['Survived']].values

x_test = preprocessing(dftest_raw).values
y_test = dftest_raw[['Survived']].values

# print("x_train.shape =", x_train.shape )
# print("x_test.shape =", x_test.shape )

# print("y_train.shape =", y_train.shape )
# print("y_test.shape =", y_test.shape )

dl_train = DataLoader(TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).float()),shuffle = True, batch_size = 8)
dl_valid = DataLoader(TensorDataset(torch.tensor(x_test).float(),torch.tensor(y_test).float()),shuffle = False, batch_size = 8)

def create_net():
    net = nn.Sequential()
    net.add_module("linear1",nn.Linear(15,20))
    net.add_module("relu1",nn.ReLU())
    net.add_module("linear2",nn.Linear(20,15))
    net.add_module("relu2",nn.ReLU())
    net.add_module("linear3",nn.Linear(15,1))
    net.add_module("sigmoid",nn.Sigmoid())
    return net
    
net = create_net()