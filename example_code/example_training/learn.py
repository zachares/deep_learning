import sys
import numpy as np
import pickle
import random
import torch
import yaml

from dataloader import CustomDataset
from deep_learning.dataset_manager.data_loading_utils import init_dataloader
from deep_learning.losses import LossFactory
from deep_learning.metrics import EvalMetricFactory
from deep_learning.models_manager.model_utils import init_and_load_models
from deep_learning.train_nn_models import train_nn_models
from example_models import ModelFactory
from training_utils import ProjectLossFactory, ProjectEvalMetricFactory


def learning_main(config_file=None):
    """ Trains a neural network according to the specifications in
        the config file example_learning_config.yml
    """
    # Loads the configuration file for this training program
    config_file = config_file or sys.argv[1]
    with open(config_file, "r") as ymlfile:
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
    idx_dict_path = cfg['dataloading_params']['idx_dict_path']
    if idx_dict_path is not None:
        with open(idx_dict_path, 'rb') as f:
            idx_dict = pickle.load(f)
        print("Loaded Train Val split dictionary from path: {}".format(idx_dict_path))
    else:
        idx_dict = None

    data_loader = init_dataloader(cfg, CustomDataset(cfg, idx_dict=idx_dict))

    # Loads the neural network architectures from example_models.py
    # specified in example_learning_config.yml into a dictionary
    model_dict = init_and_load_models(ModelFactory, cfg['info_flow'], device)

    combined_loss_factory = {}
    for member in (list(LossFactory) + list(ProjectLossFactory)):
        combined_loss_factory[member.name] = member.value

    combined_eval_factory = {}
    for member in (list(EvalMetricFactory) + list(ProjectEvalMetricFactory)):
        combined_eval_factory[member.name] = member.value

    # Trains the neural networks in the model dictionary using the
    # specified hardware and data loader objects
    train_nn_models(
        cfg,
        model_dict,
        combined_loss_factory,
        combined_eval_factory,
        data_loader,
        device
    )

if __name__ == "__main__":
	learning_main()
