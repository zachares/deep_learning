import enum
import datetime
import numpy as np
import os
import pickle
import time
import torch
from torch_geometric.data import Batch as GraphBatch
from tqdm import tqdm
from typing import Dict
from types import FunctionType
import yaml

from deep_learning.models_manager.model_wrappers import ModelWrapper
from deep_learning.logger import Logger
from deep_learning.trainer import Trainer

# TODO git hash to know the state of the code when experiments are run
def save_as_yml(name : str, dictionary : dict, save_dir : str):
    """ Saves a dictionary as a yaml file """
    print("Saving ", name, " to: ", save_dir + name + ".yaml")
    with open(save_dir + name + ".yaml", 'w') as ymlfile2:
        yaml.dump(dictionary, ymlfile2)


def save_as_pkl(name : str, dictionary : dict, save_dir : str):
    """ Saves a dictionary as a pickle file """
    print("Saving ", name, " to: ", save_dir + name + ".pkl")
    with open(save_dir + name + '.pkl', 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)


def train_nn_models(
    cfg : dict,
    model_dict : Dict[str, ModelWrapper],
    loss_factory : Dict[str, FunctionType],
    eval_factory : Dict[str, FunctionType],
    data_loader : torch.utils.data.DataLoader,
    device : torch.device
):
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
    var = input("\nRun code in debugging mode? If yes, no Results will be "
                "saved.[y/n]: ")
    if var == "y":
        logging_flag = False
        print("Currently Debugging")
        torch.autograd.set_detect_anomaly(True)
    elif var == "n":
        logging_flag = True
        print("Logging results of training")
    else:
        raise Exception("Sorry, {} is not a valid input for".format(var)
                         + "determine whether to run in debugging mode")

    var = input("\nTrain models without saving?[y/n]: ")
    if var == "y":
        save_model_flag = False
    elif var == "n":
        save_model_flag = True
    else:
        raise Exception("Sorry, {} is not a valid input for".format(var)
                        + "determine whether to save models")

    if save_model_flag or logging_flag:
        var = input("\nEvery how many epochs would you like to test the"
                    " model on the validation set and/or save it?[1,2,...,1000,...,inf]:")
        save_val_interval = int(var)
        print("Validating and saving every ", save_val_interval, " epochs")

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
        run_log_dir = "{}{}_{}_{}/".format(
            log_dir,
            date,
            load_cfg['run_tracker']['run'],
            cfg['logging_params']['run_notes']
        )
        # creating run logging directory if it does not already exist
        if not os.path.isdir(run_log_dir):
            os.mkdir(run_log_dir)
        if logging_flag:
            print(f"To view training statistics run: tensorboard --logdir={run_log_dir}")
            logger = Logger(run_log_dir)
        print("\nLOGGING AND MODEL SAVING DIRECTORY: ", run_log_dir)
    else:
        save_val_interval = np.inf

    #####################################################
    #### Setting up Trainer instance to train models  ###
    #####################################################
    if logging_flag:
        log_dict = logger.logging_dict
    else:
        log_dict = {'scalar' : {}, 'image' : {}}

    trainer = Trainer(
        cfg['training_params'],
        model_dict,
        loss_factory,
        eval_factory,
        log_dict,
        cfg['info_flow'],
        device
    )
    ################
    ### Training ###
    ################
    # Counter of the total number of iterations / updates performed
    global_cnt = 0
    # Counter of the total number of validation iterations performed
    val_cnt = 0
    # Counter of the number of epochs that have passed
    i_epoch = 0

    prev_time = time.time()

    if save_model_flag or logging_flag:
        save_as_pkl(
            "val_train_split",
            data_loader.dataset.idx_dict,
            save_dir = run_log_dir
        )
        save_as_yml("metadata", cfg, save_dir=run_log_dir)
        if save_model_flag:
            trainer.save(i_epoch, run_log_dir)

    best_val_metric = {}
    for model_key in model_dict.keys():
        best_val_metric[model_key] = np.inf
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
        i_iter = 0
        num_nodes = 0
        graphs = []
        with tqdm(data_loader, unit=" batch") as tepoch:
            for sample_batched in tepoch:
                tepoch.set_description(f"Epoch {i_epoch}")
                # useful if you are training using a curriculum
                sample_batched['epoch'] = torch.from_numpy(np.array([[i_epoch]])).float()
                sample_batched['iteration'] = torch.from_numpy(np.array([[i_iter]])).float()
                if type(sample_batched) != dict:
                    num_nodes += sample_batched.num_nodes
                    if num_nodes > 25000:
                        trainer.train(GraphBatch.from_data_list(graphs).to_dict())
                        graphs = sample_batched.to_data_list()
                        num_nodes = sample_batched.num_nodes
                    else:
                        graphs += sample_batched.to_data_list()
                else:
                    trainer.train(sample_batched)
                # logging step
                if logging_flag:
                    logger.log_results(global_cnt, 'train/', save_image=True)
                    tepoch.set_postfix(**logger.get_mean_dict())
                global_cnt += 1
                i_iter += 1
        if logging_flag:
            logger.log_means()
        ##################
        ### Validation ###
        ##################
        # performed at the end of each epoch
        if ((i_epoch + 1) % save_val_interval) == 0 or i_epoch == 0:
            print("Calculating validation results after #{} epochs".format(i_epoch))
            # setting dataloader to load from the validation set
            data_loader.dataset.val_bool = True
            data_loader.sampler.indices = range(len(data_loader.dataset))
            i_iter = 0
            with tqdm(data_loader, unit=" batch") as vepoch:
                for sample_batched in vepoch:
                    sample_batched['epoch'] = torch.from_numpy(np.array([[i_epoch]])).float()
                    sample_batched['iteration'] = torch.from_numpy(np.array([[i_iter]])).float()
                    if type(sample_batched) != dict:
                        sample_batched = sample_batched.to_dict()
                    # evaluation step
                    trainer.eval(sample_batched)
                    # logging step
                    if logging_flag:
                        logger.log_results(val_cnt, 'val/', save_image=True)
                        vepoch.set_postfix(**logger.get_mean_dict())
                    val_cnt += 1
                    i_iter += 1
            # logging step
            if logging_flag and save_model_flag:
                #checkpointing code
                for model_key in model_dict.keys():
                    if 'checkpointing_metric' in cfg["info_flow"][model_key]:
                        checkpointing_metric = cfg["info_flow"][model_key]['checkpointing_metric']
                        current_best = best_val_metric[model_key]
                        current_metric = logger.get_mean_dict()[f"{model_key}_{checkpointing_metric}"]
                        if current_metric < current_best:
                            cfg["info_flow"][model_key]["model_dir"] = run_log_dir
                            cfg["info_flow"][model_key]["epoch"] = i_epoch + 1
                            best_val_metric[model_key] = current_metric
                            save_as_yml("metadata", cfg, save_dir=run_log_dir)
                            trainer.save(i_epoch + 1, run_log_dir)
            if logging_flag:
                logger.log_means()
        # saving models at specified epoch interval ####
        if save_model_flag and ((i_epoch + 1) % save_val_interval) == 0:
            trainer.save(i_epoch + 1, run_log_dir)
