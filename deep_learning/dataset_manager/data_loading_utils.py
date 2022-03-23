import numpy as np
import torch
from torch_geometric.loader import DataLoader as GraphLoader
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Dict


def np_dict2torch_dict(dict_np : Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """ Creates a new dictionary of torch.tensors 'dict_torch' using the
        keys and values (numpy arrays) from an input dictionary 'dict_np'
        and returns the new dictionary 'dict_torch'
    """
    dict_torch = dict()
    for k, v in dict_np.items():
        if "padding_mask" in k:
            dict_torch[k] = torch.from_numpy(v).bool()
        elif k[-3:] == "idx":
            dict_torch[k] = torch.from_numpy(v).long()
        else:
            dict_torch[k] = torch.from_numpy(v).float()
    return dict_torch


def init_dataloader(
    cfg : dict,
    dataset : torch.utils.data.Dataset
) -> torch.utils.data.DataLoader:
    """ Initializes a Custom_Dataloader instance and then using it as
        an initializing argument initializes a Dataloader instance. Then
        returns the Dataloader instance

        Args:
            cfg: a dictionary with the initialization parameters for
            the Custom_Dataloader and Dataloader
            Custom_DataLoader: a child class of torch.utils.data.Dataset
            which is custom to the project which is used to load data
            during training and evaluation

        Returns:
            a Dataloader instance that will be used to load random batches
            of the dataset during training
    """
    batch_size = cfg['dataloading_params']['batch_size']
    num_workers = cfg['dataloading_params']['num_workers']
    sampler = SubsetRandomSampler(range(dataset.train_length))
    # Loading the dataset
    if (
        'graph_dataset' in cfg['dataloading_params']
        and cfg['dataloading_params']['graph_dataset'] == 1
    ):
        print("Graph Dataset Created")
        data_loader = GraphLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )
    else:
        print("Dataset Created")
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True
        )
    return data_loader
