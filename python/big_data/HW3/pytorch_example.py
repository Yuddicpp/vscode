# import libraries
import argparse
import numpy as np
import torch
from utils import *

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
epochs = None


# load data
train_image, train_label, test_image, test_label = LoadData(args.num_classes, args.num_samples_train, args.num_samples_test, args.seed)
# note: you should use train_image, train_label for training, apply the model to test_image to get predictions and use test_label to evaluate the predictions 

# train model using train_image and train_label
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    ### Your Code Here ###
    loss = None
    
    loss.backward()
    optimizer.step()
  
# get predictions on test_image
model.eval()
with torch.no_grad():
    ### Your Code Here ###
    pred = None
    
# evaluation
print("Test Accuracy:", np.mean(1.0 * (pred == test_label)))
# note that you should not use test_label elsewhere





