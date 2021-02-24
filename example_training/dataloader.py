import torch
from torch.utils.data import Dataset
import numpy as np
import copy
import time
import os
import sys

sys.path.append("../cifar_10_dataset")

from load_cifar_10 import load_cifar_10_data

import utils_sl as sl


class CustomDataset(Dataset):
    """ A class used to allocate and load the CIFAR-10 data set composed
        of 32x32 RGB images which categories into 10 different classes 
        based on the object or animal in the image.   

        Note: since the data set is small enough to load into RAM, we 
        can store it all in a NumPy array during initialization.

        Attributes:
            val_ratio: the ratio of data allocated to the validation
            set from the full data set (float)

            dev_ratio: the ratio of data allocated to the development
            set from the full data set. A development set is a very 
            small data set used just to debug code in the training 
            pipeline. (float)

            val_bool: a boolean indicator of whether the instance 
            should load data from the validation set

            dev_bool: a boolean indicator of whether the instance should
            load data from the development set

            train_length: the number of points in the training set (int)

            val_length: the number of points in the validation set (int)

            dev_length: the number of points in the development set (int)

            idx_dict: a dictionary containing three dictionaries 
            (idx_dict['val'], idx_dict['train'], idx_dict['dev']) 
            one for each data set (training, validation, development) 
            which map from an index in the training, validation or development set 
            to its corresponding index in the np.ndarrays ,self.cifar_train_data
            and self.cifar_train_labels, which is a larger array containing
            the data points for all three data sets.

            cifar_train_data: a np.ndarray with all the images in the 
            CIFAR-10 training set

            cifar_train_labels: a np.npdarray with indexes for the class
            allocation of each training image.
    """
    def __init__(self, 
                 cfg: dict,
                 idx_dict: dict = None):
        """ Inits CustomDataSet

            All arguments are explained above in the attributes section 
            above except cfg. cfg is a dictionary containing the 
            generation parameters for the data set i.e. val_ratio, the 
            path where the csvs are stored etc.
        """

        self.val_ratio = cfg['training_params']['val_ratio']
        self.use_dev = cfg['training_params']['use_dev']
        self.val_bool = False
        self.dev_bool = True if self.use_dev else False
        self.train_length = 0
        self.val_length = 0
        self.dev_length = 0

        (cifar_train_data,
         cifar_train_filenames,
         cifar_train_labels,
         cifar_test_data,
         cifar_test_filenames,
         cifar_test_labels,
         cifar_label_names) = load_cifar_10_data(cfg['dataset_path'])

        self.cifar_train_data = cifar_train_data
        self.cifar_train_labels = cifar_train_labels

        if idx_dict == None:
            print("Dataset path: ", cfg['dataset_path'])
            print("Starting Train Val Split")

            self.idx_dict = {}
            self.idx_dict['val'] = {}
            self.idx_dict['train'] = {}
            self.idx_dict['dev'] = {}
            self.idx_dict['idx_list'] = []

            self.idx_dict['idx_list'] = range(cifar_train_data.shape[0])

            self.dev_ratio = (cfg['training_params']['dev_num']
                              / len(self.idx_dict['idx_list']))

            for idx in self.idx_dict['idx_list']:
                # if train_val_bool = 1 is train, 0 is val
                train_val_bool = np.random.binomial(1, 1 - self.val_ratio, 1)
                # if dev_bool = 1 is dev
                dev_bool = np.random.binomial(1, 1 - self.dev_ratio, 1) 

                if train_val_bool == 1:
                    self.idx_dict['train'][self.train_length] = idx
                    self.train_length += 1
                else:
                    self.idx_dict['val'][self.val_length] = idx
                    self.val_length += 1

                if dev_bool == 0:
                    self.idx_dict['dev'][self.dev_length] = idx
                    self.dev_length += 1

        else:
            self.idx_dict = idx_dict
            print("Dataset path: ", 
                  self.idx_dict['generation_parameters']['dataset_path'])
            self.train_length = len(list(self.idx_dict['train'].keys()))
            self.val_length = len(list(self.idx_dict['val'].keys()))
            self.dev_length = len(list(self.idx_dict['dev'].keys()))

        print("Total data points: ", self.train_length + self.val_length)
        print("Total training points: ", self.train_length)
        print("Total validation points: ", self.val_length)
        print("Total development points: ", self.dev_length, "\n")

    def __len__(self) -> int:
        """ Returns the number of points in the data set

            Note: the dev_bool and val_bool indicate from which data set
            the Custom_Dataloader instance loads data (validation,
            training or development) 
        """
        if self.dev_bool:
            return self.dev_length     
        elif self.val_bool:
            return self.val_length
        else:
            return self.train_length

    def __getitem__(self, idx : int) -> dict:
        """ Loads a single data point to use for training or evaluation
            of a neural network

            Args:
                idx: the index of the data point from the data set to 
                load
            
            Returns:
                a dictionary of torch.Tensors containing the inputs to
                the neural network and the labels used to calculate
                its performance at the estimation / classification task
                it is being trained to perform
        """
        if self.dev_bool:
            key_set = 'dev'
        elif self.val_bool:
            key_set = 'val'
        else:
            key_set = 'train'

        array_idx = self.idx_dict[key_set][idx]

        sample = {
            'image' : self.cifar_train_data[array_idx],
            'class_idx' : np.array(self.cifar_train_labels[array_idx]),
        }

        return sl.np_dict2torch_dict(sample)
