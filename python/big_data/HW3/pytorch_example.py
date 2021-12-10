# import libraries
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.flatten import Flatten
from utils import *
from torch.utils.data import Dataset,DataLoader,TensorDataset
import setproctitle
import matplotlib.pyplot as plt

# define settings
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=50, 
                    help='number of classes used')
parser.add_argument('--num_samples_train', type=int, default=15, 
                    help='number of samples per class used for training')
parser.add_argument('--num_samples_test', type=int, default=5, 
                    help='number of samples per class used for testing')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed')

parser.add_argument('--epoch', type=int, default=100,
                    help='Set the epoch')
parser.add_argument('--model_type', type=str, default='CONV',
                    help='Set the type of model')
parser.add_argument('--opti', type=str, default='Adam',
                    help='Set the optimizer')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Set the learning rate')
                                        
args = parser.parse_args()

# define you model, loss functions, hyperparameters, and optimizers
### Your Code Here ###

setproctitle.setproctitle("zhangsy")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Model

class Conv_Net(nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()
        self.conv_model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )
        
        self.FC_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 50),
            nn.Softmax(dim=1)



        )

    def forward(self, x):
        return self.FC_model(self.conv_model(x))

class FC_Net(nn.Module):
    def __init__(self):
        super(FC_Net,self).__init__()
        self.FC_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,50)
            # nn.Softmax(dim=1)  
        )
    
    def forward(self,x):
        return self.FC_model(x)



if args.model_type == 'CONV':
    model = Conv_Net()
elif args.model_type == 'FC':
    model = FC_Net()
model = model.to(device)
epochs = args.epoch
LR = args.lr
if args.opti == 'Adam':
    optimizer =  torch.optim.Adam(model.parameters(), lr=LR)
elif args.opti == 'SGD':
    optimizer =  torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


# load data
train_image, train_label, test_image, test_label = LoadData(args.num_classes, args.num_samples_train, args.num_samples_test, args.seed)
# note: you should use train_image, train_label for training, apply the model to test_image to get predictions and use test_label to evaluate the predictions 
train_image = train_image.reshape(-1,28,28)
test_image = test_image.reshape(-1,28,28)
train_image = train_image[:,np.newaxis,:,:]
test_image = test_image[:,np.newaxis,:,:]


# dl_train = DataLoader(TensorDataset(torch.tensor(train_image).to(torch.float32),torch.tensor(train_label).long()),shuffle = True, batch_size = 8)
# dl_valid = DataLoader(TensorDataset(torch.tensor(test_image).to(torch.float32),torch.tensor(test_label).long()),shuffle = False, batch_size = 8)


model.train()
data = torch.tensor(train_image).to(torch.float32).to(device)
label = torch.tensor(train_label).long().to(device)
test_image = torch.tensor(test_image).to(torch.float32).to(device)
# train model using train_image and train_label

loss_epoch = []
for epoch in range(epochs):
    optimizer.zero_grad()
    ### Your Code Here ###
    pred = model(data)
    loss = criterion(pred,label)

    loss.backward()
    optimizer.step()
    
    loss_epoch.append(loss.item())
    print("EPOCH: "+str(epoch)+"; Loss: "+str(loss.item()))

plt.plot(range(epochs),loss_epoch)
plt.xlabel(args.model_type)
plt.ylabel("Loss")
plt.savefig(args.model_type+"_"+str(args.epoch)+".png")

# get predictions on test_image
model.eval()
with torch.no_grad():
    ### Your Code Here ###
    pred = model(test_image).cpu().numpy()
    
# evaluation
# print(pred)
# print(test_label)
pred=np.argmax(pred, axis=1)

print("Test Accuracy:", np.mean(1.0 * pred == test_label))
# note that you should not use test_label elsewhere





