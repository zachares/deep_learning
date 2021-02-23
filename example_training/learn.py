import sys
import os
import yaml
import random
import numpy as np
from dataloader import CustomDataset
import example_models as em
from train_nn_models import train_nn_models
import utils_sl as sl
import torch

import training_utils as ut_t

def learning_main():
    """ Trains a neural network according to the specifications in 
        the config file example_learning_config.yml
    """
    # Loads the configuration file for this training program
    with open("example_learning_config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # Sets the seed for training in order to have repeatable results in 
    # spite of the sampling required of training neural networks with large
    # data sets
    seed = cfg['training_params']['seed']
    random.seed(seed)
    np.random.seed(seed)

    # Checks whether a GPU is available for training and whether the
    # configuration file specifies to use a GPU 
    use_cuda = cfg["training_params"]["use_GPU"] and torch.cuda.is_available()

    if use_cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    # Loads a reference to the specified hardware (CPU or GPU)
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    # Initiates the dataloader objects to be used during training, 
    # data_loader outputs data used for training and val_data_loader
    # outputs data used for evaluation
    data_loader = sl.init_dataloader(cfg, CustomDataset)

    # Loads the neural network architectures from example_models.py 
    # specified in example_learning_config.yml into a dictionary 
    model_dict = sl.init_and_load_models(em.get_ref_model_dict(),
                                         cfg['info_flow'],
                                         device)

    project_loss_dict, project_eval_dict = ut_t.get_project_loss_and_eval_dict()

    general_loss_dict, general_eval_dict = sl.get_loss_and_eval_dict()

    if isinstance(project_loss_dict, dict):
        loss_dict = {**general_loss_dict, **project_loss_dict}

    if isinstance(project_eval_dict, dict):
        eval_dict = {**general_eval_dict, **project_eval_dict}

    # Trains the neural networks in the model dictionary using the
    # specified hardware and data loader objects 
    train_nn_models(cfg, 
                    model_dict,
                    loss_dict,
                    eval_dict,
                    data_loader,
                    device)

if __name__ == "__main__":
	learning_main()