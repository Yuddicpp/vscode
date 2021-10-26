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
from model_sep import *
from baseline import svm_regressor
from test import *

import logging


def train_identifier(opt, device, result_path, model_path, dataloader, val_dataloader, network, optimizer, data_raw):
    
    # Save training log
    logging.basicConfig(filename=os.path.join(result_path, 'train_identifier_log.log'), level=logging.INFO)
    logging.info("Started")

    # Load pre-trained model or initialize weights
    if opt.epoch != 0:
        network.load_state_dict(torch.load(os.path.join(model_path, "INet_%d.pth" % opt.epoch)))
    else:
        network.apply(weights_init_normal)

    # Set loss function
    criterion_idy = torch.nn.CrossEntropyLoss().to(device)

    # Loss weights
    lambda_idy = 1

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_i_epochs):

        # Initialization evaluation metrics
        # rmse_error = 0.0
        # abs_error = 0.0
        accuracy = 0.0
        start_time = time.time()

        for i, batch in enumerate(dataloader):
            
            # Set model input
            cir_gt = batch["CIR"]
            label_gt = batch["Label"]
            if torch.cuda.is_available():
                cir_gt = cir_gt.cuda()
                label_gt = label_gt.to(device=device, dtype=torch.int64)  # torch.LongTensor

            # Start training
            optimizer.zero_grad()

            # Generate estimation (vector)
            label_est = network(cir_gt)

            # Loss terms
            label_gt = label_gt.squeeze()  # 0~n-1 for cross-entropy
            # label_gt = label_gt.type(torch.LongTensor)
            loss_idy = lambda_idy * criterion_idy(label_est, label_gt)

            # Total loss
            loss = loss_idy  # + loss_reg

            loss.backward()
            optimizer.step()

            # ------ log process ------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_i_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))

            # Evaluation
            with torch.no_grad():
                time_train = time.time() - start_time
                time_avg = time_train / (i + 1)
                # env label estimation
                prediction = torch.argmax(label_est, dim=1)  # (b, num_classes)
                accuracy += torch.sum(prediction==label_gt).float() / label_gt.shape[0]
                accuracy_avg = accuracy / (i + 1)

            # Print log
            sys.stdout.write(
                "\r[Train: Data Env: %s] [Model Type: Identifier%s] [Epoch: %d/%d] [Batch: %d/%d]"
                "[Sep Idy Loss: %f] [accuracy %f] [Train Time: %f, ETA: %s]"
                % (opt.dataset_env, opt.identifier_type, (epoch+1), opt.n_i_epochs, i, len(dataloader),
                loss.item(), accuracy_avg, time_avg, time_left)
            )
            logging.info(
                "\r[Train: Data Env: %s] [Model Type: Identifier%s] [Epoch: %d/%d] [Batch: %d/%d]"
                "[Sep Idy Loss: %f] [accuracy %f] [Train Time: %f, ETA: %s]"
                % (opt.dataset_env, opt.identifier_type, (epoch+1), opt.n_i_epochs, i, len(dataloader),
                loss.item(), accuracy_avg, time_avg, time_left)
            )
        
        # Update learning rate
        # lr_scheduler.step()

        # Save models if at checkpoint epoch
        if opt.checkpoint_interval != -1 and (epoch+1) % opt.checkpoint_interval == 0:
            torch.save(network.state_dict(), os.path.join(model_path, "INet_%d.pth" % (epoch+1)))
            print("Saving model of epoch %d" % (epoch+1))

        # # Illustrate results on test set if at sample interval epoch (sample_interval % checkpoint_interval = 0)
        if (epoch+1) % opt.sample_interval == 0:
            test_identifier(
                opt=opt, device=device, result_path=result_path, model_path=model_path, 
                dataloader=val_dataloader, network=network, epoch=(epoch+1), data_raw=data_raw
            )  # epoch for val and opt.test_epoch for test
            

def train_estimator(opt, device, result_path, model_path, dataloader, val_dataloader, network, optimizer, data_raw):
    
    # Save training log
    logging.basicConfig(filename=os.path.join(result_path, 'training_estimator_log.log'), level=logging.INFO)
    logging.info("Started")

    # Load pre-trained model or initialize weights
    if opt.epoch != 0:
        network.load_state_dict(torch.load(os.path.join(model_path, "ENet_%d.pth" % opt.epoch)))
    else:
        network.apply(weights_init_normal)

    # Set loss function
    criterion_reg = torch.nn.L1Loss().to(device)
    # criterion_idy = torch.nn.CrossEntropyLoss().to(device)

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_e_epochs):

        # Initialization evaluation metrics
        rmse_error = 0.0
        abs_error = 0.0
        start_time = time.time()

        for i, batch in enumerate(dataloader):
            
            # Set model input
            cir_gt = batch["CIR"]
            err_gt = batch["Err"]
            label_gt = batch["Label"]
            if torch.cuda.is_available():
                cir_gt = cir_gt.cuda()
                err_gt = err_gt.cuda()
                # label_gt_t = label_gt.to(device=device, dtype=torch.int64)  # torch.LongTensor
                # label_gt_t = label_gt.cuda()

            # Start training
            optimizer.zero_grad()

            # Generate estimations
            # label_est, env_latent = enet(cir_gt)
            # transfer label to one-hot form
            label_input = label_gt.numpy().astype('int64')
            # print(label_input[0:5])
            label_cat = to_categorical(
                label_input, num_columns=opt.num_classes
            )
            # print(label_cat[0:5])
            label_cat = torch.tensor(label_cat)
            label_cat = label_cat.to(device=device, dtype=torch.int64)
            # print("testing label format: ", label_gt_t[0:5], label_cat[0:5])
            err_mu, err_sigma, err_sri = network(label_cat, cir_gt)
            # use err_mu as est and others for loss term

            # Loss term (require modification)
            loss_reg = criterion_reg(err_mu, err_gt)
            epsilon = 0.1
            kl_div = 0.5 * torch.sum(
                ((2 * err_sigma).exp() + (err_mu - err_gt) ** 2)/(epsilon ** 2) - 1 - 2 * err_sigma, dim=1
            )
            loss = kl_div.mean()  # account for batch size b
            # print("testing loss terms: ", loss_reg, loss)

            loss.backward()
            optimizer.step()

            # ------ log process ------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_e_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))

            # Evaluation
            with torch.no_grad():
                # range error estimation
                rmse_error += (torch.mean((err_mu - err_gt) ** 2)) ** 0.5
                abs_error += torch.mean(torch.abs(err_mu - err_gt))
                time_train = time.time() - start_time
                rmse_avg = rmse_error / (i + 1)
                abs_avg = abs_error / (i + 1)
                time_avg = time_train / (i + 1)

            # Print log
            sys.stdout.write(
                "\r[Train Data Env: %s] [Model Type: Estimator%s] [Epoch: %d/%d] [Batch: %d/%d]"
                "[Reg Loss: %f] [Error: rmse %f, abs %f] [Train Time: %f, ETA: %s]"
                % (opt.dataset_env, opt.estimator_type, (epoch+1), opt.n_e_epochs, i, len(dataloader),
                loss.item(), rmse_avg, abs_avg, time_avg, time_left)
            )
            logging.info(
                "\r[Train Data Env: %s] [Model Type: Estimator%s] [Epoch: %d/%d] [Batch: %d/%d]"
                "[Reg Loss: %f] [Error: rmse %f, abs %f] [Train Time: %f, ETA: %s]"
                % (opt.dataset_env, opt.estimator_type, (epoch+1), opt.n_e_epochs, i, len(dataloader),
                loss.item(), rmse_avg, abs_avg, time_avg, time_left)
            )

        # Save models if at checkpoint epoch
        if opt.checkpoint_interval != -1 and (epoch+1) % opt.checkpoint_interval == 0:
            torch.save(network.state_dict(), os.path.join(model_path, "ENet_%d.pth" % (epoch+1)))
            print("Saving model of epoch %d" % (epoch+1))

        # # Illustrate results on test set if at sample interval epoch
        if (epoch+1) % opt.sample_interval == 0:
            test_estimator(
                opt=opt, device=device, result_path=result_path, model_path=model_path, 
                dataloader=val_dataloader, network=network, epoch=(epoch+1), data_raw=data_raw
            )  # epoch for val and opt.test_epoch for test

