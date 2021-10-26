import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as FloatTensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from utils import *
from dataset import *
from model import *
from baseline import svm_regressor
from train import *
from test import *

import setproctitle
import logging


setproctitle.setproctitle("UWB_EM")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Get arguments
parser = argparse.ArgumentParser()
parser = get_args(parser)
opt = parser.parse_args()
print(opt)

# Initialize network
if opt.dataset_name == 'zenodo':
    opt.cir_len = 157
    if opt.dataset_env == 'room_full':
        opt.num_classes = 5
    elif opt.dataset_env == 'obstacle_full':
        opt.num_classes = 10
    elif opt.dataset_env == 'nlos':
        opt.num_classes = 2
    elif opt.dataset_env == 'room_part':
        opt.num_classes = 3
    elif opt.data_env == 'obstacle_part':
        opt.num_classes = 4
elif opt.dataset_name == 'ewine':
    opt.cir_len = 152
    opt.dataset_env = 'nlos'
    opt.num_classes = 2

# select neural module arrangement method
if opt.net_ablation == 'loop':
    Network = EMNet(
        cir_len=opt.cir_len, num_classes=opt.num_classes, env_dim=opt.env_dim, 
        filters=opt.filters, enet_type=opt.identifier_type, mnet_type=opt.regressor_type
    ).to(device)
elif opt.net_ablation == 'loops':
    Network = EMNetLoop(
        cir_len=opt.cir_len, num_classes=opt.num_classes, env_dim=opt.env_dim, 
        filters=opt.filters, enet_type=opt.identifier_type, mnet_type=opt.regressor_type
    ).to(device)
# elif opt.net_ablation == 'detach':
#     ENet = Identifier(cir_len=opt.cir_len, num_classes=opt.num_classes, env_dim=opt.env_dim,
#         filters=opt.filters, enet_type=opt.identifier_type).to(device)
#     MNet = Regressor(cir_len=opt.cir_len, num_classes=opt.num_classes, env_dim=opt.env_dim,
#         filters=opt.filters, mnet_type=opt.regressor_type).to(device)
else:
    raise ValueError("Unknown network arrangement, choices: loop, loops, detach.")

# Create sample and checkpoint directories
model_path = "./saved_models_%s/data_%s_%s_mode_%s/enet%d_mnet%d" % (opt.net_ablation, opt.dataset_name, opt.dataset_env, opt.mode, opt.identifier_type, opt.regressor_type)
train_path = "./saved_results_%s/data_%s_%s_mode_%s/enet%d_mnet%d" % (opt.net_ablation, opt.dataset_name, opt.dataset_env, opt.mode, opt.identifier_type, opt.regressor_type)
os.makedirs(model_path, exist_ok=True)
os.makedirs(train_path, exist_ok=True)
test_path = "./saved_results_%s/test/data_%s_%s_mode_%s/enet%d_mnet%d" % (opt.net_ablation, opt.dataset_name, opt.dataset_env, opt.mode, opt.identifier_type, opt.regressor_type)
os.makedirs(test_path, exist_ok=True)

# Optimizers
# if opt.net_ablation == 'detach':
#     optimizer = torch.optim.Adam(
#         itertools.chain(ENet.parameters(), MNet.parameters()),
#         lr=opt.lr,
#         betas=(opt.b1, opt.b2)
#     )
# else:
optimizer = torch.optim.Adam(
    Network.parameters(),
    lr=opt.lr,
    betas=(opt.b1, opt.b2)
)


# # Learning rate update schedulers (not used)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
# )

# Get data
print("Loading dataset from %s_%s for training." % (opt.dataset_name, opt.dataset_env))
if opt.dataset_name == 'zenodo':
    root = './data/data_zenodo/dataset.pkl'
elif opt.dataset_name == 'ewine':
    filepaths = ['./data/data_ewine/dataset1/tag_room0.csv',
                 './data/data_ewine/dataset1/tag_room1.csv',
                 './data/data_ewine/dataset2/tag_room0.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part0.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part1.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part2.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part3.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part4.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part5.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part6.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part7.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part8.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part9.csv']
    root = filepaths

data_train, data_test = assign_train_test(opt, root)

# Configure dataloaders
dataloader_train = DataLoader(
    dataset=UWBDataset(data_train),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=8,
)

dataloader_test = DataLoader(
    dataset=UWBDataset(data_test),
    batch_size=500,
    shuffle=True,
    num_workers=1,
)


# ------------- Training --------------

data = data_train, data_test
# if opt.net_ablation == 'detach':
#     train_gem(
#         opt, device=device, tensor=Tensor, result_path=train_path, model_path=model_path, 
#         dataloader=dataloader_train, val_dataloader=dataloader_test,
#         optimizer=optimizer, enet=ENet, mnet=MNet, data_raw=data
#     )
# else:
train_gem(
    opt, device=device, tensor=Tensor, result_path=train_path, model_path=model_path, 
    dataloader=dataloader_train, val_dataloader=dataloader_test,
    optimizer=optimizer, network=Network, data_raw=data
)

# ------------- Testing --------------

# if opt.net_ablation == 'detach':
#     test_gem(
#         opt=opt, device=device, tensor=Tensor, result_path=test_path, model_path=model_path, 
#         dataloader=dataloader_test, enet=ENet, mnet=MNet, epoch=opt.test_epoch, daat_raw=data
#     )  # epoch for val and opt.test_epoch for test
# else:
test_gem(
    opt=opt, device=device, tensor=Tensor, result_path=test_path, model_path=model_path, 
    dataloader=dataloader_test, network=Network, epoch=opt.test_epoch, daat_raw=data
)  # epoch for val and opt.test_epoch for test
