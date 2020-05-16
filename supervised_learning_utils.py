import h5py
import torch
import numpy as np
import time
import datetime
import os
import sys
import yaml
import copy
import random
import pickle

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

from dataloader import *
class Proto_Metric(object):	
	def __init__(self, metric_function, metric_names):
		self.metric = metric_function
		self.metric_names = tuple(metric_names)

	def measure(self, input_tuple, logging_dict, label, args_tuple = None):
		net_est, target = input_tuple

		measurements = self.metric(net_est, target)

		for i, metric_name in enumerate(self.metric_names):
			if type(measurements) == tuple:
				measurement = measurements[i]
			else:
				measurement = measurements

			logging_dict['scalar'][label + "/" + metric_name] = measurement.mean().item()

class Proto_Loss(object):
	def __init__(self, loss_function):
		self.loss_function = loss_function

	def loss(self, input_tuple, logging_dict, weight, label, args_tuple = None):
		net_est, target = input_tuple

		loss = weight * self.loss_function(net_est, target).mean()

		logging_dict['scalar'][label + "/loss"] = loss.mean().item()

		return loss

class Proto_MultiStep_Loss(object):
	def __init__(self, loss_function):
		self.loss_function = loss_function

	def loss(self, input_tuple, logging_dict, weight, label, args_tuple = None):
		net_ests, targets = input_tuple[0]
		loss = torch.zeros(1).float().to(net_ests.device)

		for i in range(net_est.size(0)):
			net_est = net_ests[i]
			target_est = targets[i]

			loss += weight * self.loss_function(net_est, target).mean()

		logging_dict['scalar'][label + "/loss"] = loss.item() / net_est.size(0)

		return loss

def init_dataloader(cfg, device, idx_dict_path = None):
    ###############################################
    ########## Loading dataloader parameters ######
    ###############################################
    batch_size = cfg['dataloading_params']['batch_size']
    num_workers = cfg['dataloading_params']['num_workers']
    run_mode = cfg['training_params']['run_mode'] 
    val_ratio = cfg['training_params']['val_ratio']

    #### loading previous val_train split to continue training a model
    if idx_dict_path is not None:
        with open(idx_dict_path, 'rb') as f:
            idx_dict = pickle.load(f)

        print("Loaded Train Val split dictionary from path: " + idx_dict_path)
    else:
        idx_dict = None

    dataset = H5_DataLoader(cfg, idx_dict = idx_dict, device = device, transform=transforms.Compose([ToTensor(device = device)]))

    if val_ratio == 0:
        print("No validation set")

    if run_mode == 0:
        train_sampler = SubsetRandomSampler(range(dataset.dev_length))
        dataset.dev_bool = True
    else:
        train_sampler = SubsetRandomSampler(range(dataset.train_length))

    data_loader = DataLoader(dataset, batch_size= batch_size, num_workers=num_workers, sampler=train_sampler, pin_memory = True) 

    val_data_loader = None  

    if val_ratio != 0:
        val_sampler = SubsetRandomSampler(range(dataset.val_length))
        val_dataset = copy.deepcopy(dataset)
        val_dataset.val_bool = True
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, sampler= val_sampler, pin_memory = True)

    return data_loader, val_data_loader, dataset.idx_dict