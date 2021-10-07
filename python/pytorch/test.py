
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import copy
import torch 
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset,sampler



def create_net_4C():
    net = nn.Sequential()
    net.add_module("linear1",nn.Linear(12,50))
    net.add_module("relu1",nn.ReLU())
    net.add_module("linear2",nn.Linear(50,10))
    net.add_module("relu2",nn.ReLU())
    net.add_module("linear3",nn.Linear(10,1))
    return net

def question_4C(data_raw):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    data = data_raw[data_raw['群类别']<=4]
    x = data.iloc[:,2:14].to_numpy()
    labels = data[['群类别']].to_numpy()

    split_num = int(x.shape[0] * 0.8)
    index_list = list(range(x.shape[0]))
    train_idx, valid_idx = index_list[:split_num], index_list[split_num:]

    tr_sampler = sampler.SubsetRandomSampler(train_idx)
    val_sampler = sampler.SubsetRandomSampler(valid_idx)

    train = DataLoader(TensorDataset(torch.tensor(x).float(),torch.tensor(labels).float()), batch_size=8,sampler=tr_sampler)
    test = DataLoader(TensorDataset(torch.tensor(x).float(),torch.tensor(labels).float()), batch_size=8,sampler=val_sampler)
    
    net = create_net_4C()
    net.to(device) 
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(params=net.parameters(),lr = 1e-7)
    net.train()
    for i in range(10000):
        loss_sum = 0

        for step, (features,labels) in enumerate(train, 1):
            features = features.to(device) # 移动数据到cuda
            labels = labels.to(device) # 或者 labels = labels.cuda() if torch.cuda.is_available() else labels
            optimizer.zero_grad()

            predictions = net(features)
            loss = loss_func(predictions,labels)
            # loss_array = np.vstack((loss_array,(predictions-labels).detach().numpy()))

            loss.backward()
            optimizer.step()

            loss_sum+=loss.item()
        print("Loss:"+str(loss_sum/step))
    
    # print(net.state_dict())

    net.eval()
    for step, (features,labels) in enumerate(test, 1):
        with torch.no_grad():
            features = features.to(device) # 移动数据到cuda
            labels = labels.to(device) # 或者 labels = labels.cuda() if torch.cuda.is_available() else labels
            predictions = net(features)
            print(predictions)
            loss = loss_func(predictions,labels)
            # loss_array = np.vstack((loss_array,(predictions-labels).detach().numpy()))

            # loss.backward()
            # optimizer.step()

            loss_sum+=loss.item()
    print("Loss:"+str(loss_sum/step))


data = pd.read_excel('data.xlsx')

question_4C(data)