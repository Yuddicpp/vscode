# import libraries
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from torch.utils.data import Dataset,DataLoader,TensorDataset

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
args = parser.parse_args()

# define you model, loss functions, hyperparameters, and optimizers
### Your Code Here ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 50)

    def forward(self, x):
        x = self.conv1(x) #26*26
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x) #24*24
        x = self.bn1(x)
        x = F.relu(x)
        
        x = F.max_pool2d(x, 2) #12*12

        x = self.conv3(x) #10*10
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv4(x) #8*8
        x = self.bn2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2) #4*4

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output




model = Net()
model = model.to(device)
epochs = 1000
LR = 0.01
optimizer =  torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


# load data
train_image, train_label, test_image, test_label = LoadData(args.num_classes, args.num_samples_train, args.num_samples_test, args.seed)
# note: you should use train_image, train_label for training, apply the model to test_image to get predictions and use test_label to evaluate the predictions 
train_image = train_image.reshape(-1,28,28)
test_image = test_image.reshape(-1,28,28)
train_image = train_image[:,np.newaxis,:,:]
test_image = test_image[:,np.newaxis,:,:]
# dl_train = DataLoader(TensorDataset(torch.tensor(train_image).to(torch.float32).to(device),torch.tensor(train_label).long().to(device)),shuffle = True, batch_size = 8)
# dl_valid = DataLoader(TensorDataset(torch.tensor(test_image).to(torch.float32).to(device),torch.tensor(test_label).long().to(device)),shuffle = False, batch_size = 8)


# train model using train_image and train_label
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    ### Your Code Here ###
    pred = model(torch.tensor(train_image).to(torch.float32).to(device))
    loss = criterion(pred,torch.tensor(train_label).long().to(device))

    loss.backward()
    optimizer.step()
    print("EPOCH: "+str(epoch)+"; Loss: "+str(loss.item()))
  
# get predictions on test_image
model.eval()
with torch.no_grad():
    ### Your Code Here ###
    pred = model(torch.tensor(test_image).to(torch.float32).to(device))
    pred = torch.argmax(pred,dim=1).cpu().numpy()
    
# evaluation
print(test_label)
print(pred)
print("Test Accuracy:", np.mean(1.0 * (pred == test_label)))
# note that you should not use test_label elsewhere





