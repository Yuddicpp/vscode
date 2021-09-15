---
title: 基于全连接神经网络的MNIST手写数字识别
date: 2020-05-04 11:30:25
tags: MNIST
categories: Python
cover: /img/MNIST/top_cover.jpg
top_img: /img/MNIST/top_cover.jpg
description: 基于pytorch框架搭建全连接神经网络来进行手写数字图片的训练与测试
---

# 导入数据

```python
#随机梯度下降法
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.datasets import MNIST # 导入 pytorch 内置的 mnist 数据
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

def data_tf(x):
	x = np.array(x,dtype = 'float32')/255
	x = (x-0.5)/0.5
	x = x.reshape((-1,))
	x = torch.from_numpy(x)
	return x

train_set = MNIST('./data', train=True, transform=data_tf, download=True) # 重新载入数据集，申明定义的数据变换
test_set = MNIST('./data', train=False, transform=data_tf, download=True)


train_data = DataLoader(train_set,batch_size=64,shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)


```


# 定义损失函数以及网络等

```python
#损失函数
criterion = nn.CrossEntropyLoss()
def sgd_update(parameters,lr):
	for param in parameters:
		param.data = param.data - lr*param.grad.data

net = nn.Sequential(
		nn.Linear(784,600),
		nn.ReLU(),
		nn.Linear(600,400),
                nn.ReLU(),
		nn.Linear(400,200),
                nn.ReLU(),
		nn.Linear(200,100),
		nn.ReLU(),
		nn.Linear(100,10)
	)
optimzier = torch.optim.SGD(net.parameters(), 1e-2)
losses1 = []
idx = 0
```

# 开始训练
```python
#加载之前已经训练好的模型参数文件
net.load_state_dict(torch.load('mnist.pth'))

start = time.time()
for e in range(100):
	train_loss = 0
	for im ,label in train_data:
		im.float().requires_grad_()
		label.float().requires_grad_()
		out = net(im)
		loss = criterion(out,label)
		net.zero_grad()
		loss.backward()
		sgd_update(net.parameters(),1e-2)
		# print(loss.data.numpy())
		train_loss+=loss.data.numpy()
		if(idx%30==0):
			losses1.append(loss.data.numpy())
		idx+=1
	print('epoch :{},Train_loss:{:.10f}'.format(e,train_loss/len(train_data)))
torch.save(net.state_dict(),'mnist.pth')
end = time.time()
print('time:{:.5f}'.format(end-start))
```



