import os
import sys
import torch
import yaml
import numpy as np
import time
import datetime

from trainer import Trainer
from logger import Logger
import utils_sl as sl

from typing import Dict
from types import FunctionType
from models_modules import ModelWrapper

# TODO git hash to know the state of the code when experiments are run
def train_nn_models(cfg : dict, 
                    model_dict : Dict[str, ModelWrapper],
                    loss_dict : Dict[str, FunctionType],
                    eval_dict : Dict[str, FunctionType],
                    data_loader : torch.utils.data.DataLoader,
                    device : torch.device):
    """ Trains models composed of neural networks, saves the models and 
        logs the results of the training at specified intervals

        Args:
            cfg: a dictionary containing the hyperparameters and loss 
            functions to use to train the models, as well as the
            location to save training results and what evaluation metrics
            to run

            model_dict: a dictionary of the models that could be trained

            loss_dict: a dictionary of the loss functions that could be
            used to train the models

            eval_dict: a dictionary of the evaluation metric that could 
            be used to evaluate training

            data_loader: an iterator which outputs random batches from
            the dataset used to train the models
            
            device: a reference to the hardware the models will be trained
            on (GPU or CPU)
        
        Raises:
            Exception: if the user inputs requested by the program do not
            conform to the set of compatible inputs
    """
    ##################################################
    ### Setting Debugging Flag and Save Model Flag ###
    ### and Setting up Logging if Required         ###
    ################################################## 
    var = input("Run code in debugging mode? If yes, no Results will be "
                "saved.[y/n]: ")
    if var == "y":
        logging_flag = False
    elif var == "n":
        logging_flag = True
    else:
        raise Exception("Sorry, {} is not a valid input for".format(var)
                         + "determine whether to run in debugging mode")

    if logging_flag:
        print("Currently Debugging")
        torch.autograd.set_detect_anomaly(True)
    else:
        print("Logging results of training")

    var = input("Train models without saving?[y/n]: ")
    if var == "y":
        save_model_flag = False
    elif var == "n":
        save_model_flag = True
    else:
        raise Exception("Sorry, {} is not a valid input for".format(var)
                        + "determine whether to save models")

    if save_model_flag or logging_flag:
        var = input("Every how many epochs would you like to test the"
                    " model on the validation set and/or save it?[1,2,...,1000,...,inf]:")
        save_val_interval = int(var)
        print("Validating and saving every ", save_val_interval, " epochs")
    else:
        save_val_interval = np.inf

    # Creating directory to save models and results in
    if logging_flag or save_model_flag:
        t_now = time.time()
        date = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d')

        # Code to keep track of runs during a day and create a 
        # unique path for logging each run
        with open("run_tracking.yml", 'r+') as ymlfile1:
            load_cfg = yaml.safe_load(ymlfile1)

        if load_cfg['run_tracker']['date'] == date:
            load_cfg['run_tracker']['run'] +=1
        else:
            print("New day of training!")
            load_cfg['run_tracker']['date'] = date
            load_cfg['run_tracker']['run'] = 0

        with open("run_tracking.yml", 'w') as ymlfile1:
            yaml.dump(load_cfg, ymlfile1)

        log_dir = cfg['logging_params']['logging_dir']

        # creating parent logging directory, if it does not already exist
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        run_log_dir = "{}{}_{}_{}/".format(log_dir, 
                                           date,
                                           load_cfg['run_tracker']['run'],
                                           cfg['logging_params']['run_notes'])

        # creating run logging directory if it does not already exist
        if not os.path.isdir(run_log_dir):
            os.mkdir(run_log_dir)

        if logging_flag: logger = Logger(run_log_dir)

        print("Logging and Model Saving Directory: ", run_log_dir)

    #####################################################
    #### Setting up Trainer instance to train models  ###
    #####################################################
    if logging_flag: 
        log_dict = logger.logging_dict
    else:
        log_dict = {'scalar' : {}, 'image' : {}}
    
    trainer = Trainer(cfg['training_params'],
                      model_dict,
                      loss_dict,
                      eval_dict,
                      log_dict,
                      cfg['info_flow'],
                      device)
    ################
    ### Training ###
    ################
    global_cnt = 0
    i_epoch = 0
    prev_time = time.time()

    if save_model_flag or logging_flag:
        sl.save_as_pkl("val_train_split", 
                       data_loader.dataset.idx_dict, 
                       save_dir = run_log_dir)

        sl.save_as_yml("learning_params", cfg, save_dir = run_log_dir)

        trainer.save(i_epoch, run_log_dir)

    for i_epoch in range(cfg['training_params']['max_training_epochs']):
        current_time = time.time()

        # Prints out the time required per epoch to the terminal
        if i_epoch != 0:
            print("Epoch took ", current_time - prev_time, " seconds")
            prev_time = time.time()

        print('Training epoch #{}...'.format(i_epoch))

        # Setting the dataloader to load from the training set
        data_loader.dataset.val_bool = False 
        data_loader.sampler.indices = range(len(data_loader.dataset))
        # print(torch.cuda.memory_summary(device))
        for i_iter, sample_batched in enumerate(data_loader):
            # useful if you are training using a curriculum
            # print(torch.cuda.memory_summary(device))
            sample_batched['epoch'] = torch.from_numpy(np.array([[i_epoch]])).float()
            sample_batched['iteration'] = torch.from_numpy(np.array([[i_iter]])).float()

            # training step
            trainer.train(sample_batched)

            global_cnt += 1
            
            # logging step
            # FIXME: magic number used to choose when to log image metrics
            if logging_flag:
                if global_cnt % 50 == 0 or global_cnt == 1:
                    logger.log_results(global_cnt, 'train/', save_image=True)
                else:
                    logger.log_results(global_cnt, 'train/')

        ##################
        ### Validation ###
        ##################
        # performed at the end of each epoch
        if ((i_epoch + 1) % save_val_interval) == 0 or i_epoch == 0:
            print("Calculating validation results after #{} epochs".format(i_epoch))

            # setting dataloader to load from the validation set
            data_loader.dataset.val_bool = True
            data_loader.sampler.indices = range(len(data_loader.dataset))

            for i_iter, sample_batched in enumerate(data_loader):
                sample_batched['epoch'] = torch.from_numpy(np.array([[i_epoch]])).float()
                sample_batched['iteration'] = torch.from_numpy(np.array([[i_iter]])).float()

                # evaluation step
                trainer.eval(sample_batched)

                # logging step
                if logging_flag: logger.log_results(global_cnt + i_iter, 'val/')

            # logging step
            if logging_flag:
                # to stop double logging of scalar results
                logger.logging_dict['scalar'] = {}
                # logging images from validation run
                logger.log_results(global_cnt + i_iter, 'val/', save_image=True)

        # saving models at specified epoch interval ####
        if save_model_flag and ((i_epoch + 1) % save_val_interval) == 0:
            trainer.save(i_epoch + 1, run_log_dir)

