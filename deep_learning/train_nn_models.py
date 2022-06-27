import enum
import datetime
import numpy as np
import os
import pickle
import time
import torch
from torch_geometric.data import Batch as GraphBatch
from tqdm import tqdm
from typing import Dict, Tuple
from types import FunctionType
import wandb
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


def make_run_logging_directory(logging_directory : str, run_name: str) -> Tuple[str, str]:
    date_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d')
    if not os.path.isdir(logging_directory):
        os.mkdir(logging_directory)
    counter = 0
    run_description = f"{date_str}_{run_name}_{counter}"
    run_log_dir = os.path.join(logging_directory,f"{run_description}/")
    while os.path.isdir(run_log_dir):
        counter += 1
        run_description = f"{date_str}_{run_name}_{counter}"
        run_log_dir = os.path.join(logging_directory,f"{run_description}/")
    os.mkdir(run_log_dir)
    return run_log_dir, run_description


def train_nn_models(
    cfg : dict,
    model_dict : Dict[str, ModelWrapper],
    loss_factory : Dict[str, FunctionType],
    eval_factory : Dict[str, FunctionType],
    data_loader : torch.utils.data.DataLoader,
    device : torch.device,
    logging_flag : bool,
    save_model_flag : bool
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
    #####################################################
    #### Setting up Trainer instance to train models  ###
    #####################################################
    assert (logging_flag and save_model_flag) or not save_model_flag
    i_epoch = 0
    prev_time = time.time()
    logger = Logger(logging_flag)
    trainer = Trainer(
        cfg['training_params'],
        model_dict,
        loss_factory,
        eval_factory,
        logger.logging_dict,
        cfg['info_flow'],
        device
    )
    if save_model_flag or logging_flag:
        run_log_dir, run_description = make_run_logging_directory(
            logging_directory=cfg['logging_params']['logging_dir'],
            run_name=cfg['logging_params']['run_name']
        )
        save_as_pkl("val_train_split", data_loader.dataset.idx_dict, save_dir=run_log_dir)
        save_as_yml("metadata", cfg, save_dir=run_log_dir)
        if logging_flag:
            wandb.init(
                config=cfg,
                project=cfg['logging_params']['wandb_project'],
                entity=cfg['logging_params']['wandb_entity'],
                name=run_description
            )
        if save_model_flag:
            trainer.save(i_epoch, run_log_dir)
            for model_key in model_dict.keys():
                assert 'checkpointing_metric' in cfg["info_flow"][model_key]
        if logging_flag and save_model_flag:
            best_val_metric = {}
            for model_key in model_dict.keys():
                best_val_metric[model_key] = np.inf
    ################
    ### Training ###
    ################
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
        logger.label = "train"
        num_nodes = 0
        graphs = []
        with tqdm(data_loader, unit=" batch") as tepoch:
            for sample_batched in tepoch:
                tepoch.set_description(f"Epoch {i_epoch}")
                if type(sample_batched) != dict:
                    if cfg['dataset_config']['max_batch_size']:
                        num_nodes += sample_batched.num_nodes
                        if num_nodes > cfg['dataset_config']['max_graph_size'] and len(graphs) > 0:
                            trainer.train(GraphBatch.from_data_list(graphs).to_dict())
                            graphs = sample_batched.to_data_list()
                            num_nodes = sample_batched.num_nodes
                        else:
                            graphs += sample_batched.to_data_list()
                    else:
                        trainer.train(sample_batched.to_dict())

                else:
                    trainer.train(sample_batched)
                logger.log_scalars()
                tepoch.set_postfix(**logger.get_mean_dict())
        logger.log_means()
        ##################
        ### Validation ###
        ##################
        # performed at the end of each epoch
        print("Calculating validation results after #{} epochs".format(i_epoch))
        # setting dataloader to load from the validation set
        data_loader.dataset.val_bool = True
        data_loader.sampler.indices = range(len(data_loader.dataset))
        logger.label = "val"
        with tqdm(data_loader, unit=" batch") as vepoch:
            for sample_batched in vepoch:
                if type(sample_batched) != dict:
                    sample_batched = sample_batched.to_dict()
                trainer.eval(sample_batched)
                logger.log_scalars()
                vepoch.set_postfix(**logger.get_mean_dict())
        if save_model_flag:
            for model_key in model_dict.keys():
                checkpointing_metric = cfg["info_flow"][model_key]['checkpointing_metric']
                current_best = best_val_metric[model_key]
                current_metric = logger.get_mean_dict()[f"{model_key}_{checkpointing_metric}"]
                if current_metric < current_best:
                    print(f"Checkpoint model {model_key}")
                    cfg["info_flow"][model_key]["model_dir"] = run_log_dir
                    cfg["info_flow"][model_key]["epoch"] = i_epoch + 1
                    best_val_metric[model_key] = current_metric
                    save_as_yml("metadata", cfg, save_dir=run_log_dir)
                    trainer.save(i_epoch + 1, run_log_dir)
        logger.log_means()
