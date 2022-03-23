from collections import OrderedDict
import enum
import yaml
import torch
from typing import Dict, Any

from deep_learning.models_manager.model_wrappers import ModelWrapper


def init_and_load_models(
    ref_model_dict : enum.Enum,
    info_flow : Dict[str, Any],
    device : torch.device
) -> Dict[str, ModelWrapper]:
    """ Initializes an instance of each model class specified by the
        dictionary infoflow which is an argument of the function. Then
        loads the weights of the model if info_flow contains a directory
        to load that model from.

        Args:
            ref_model_dict: a dictonary containing named classes of all
            the possible models that could be loaded by this function.

            info_flow: a dictonary containing instructions about which
            models to initialize and if specified where to load the model
            weights from

            device: a reference to hardware (CPU or GPU) that all the
            initialized models should perform there calculations on

        Returns:
            a dictionary containing named instances of the models that
            will be used during the run
    """
    model_dict = OrderedDict()
    for model_name in info_flow.keys():
        if info_flow[model_name]['model_dir'] is None:
            model_dict[model_name] = ref_model_dict[model_name].value(
                model_name=model_name,
                init_args=info_flow[model_name]['init_args'],
                device=device
            )
            model_dict[model_name].set_device(device)

        elif info_flow[model_name]['model_dir'][-3:] == 'pkl':
            data = torch.load(info_flow[model_name]['model_dir'])
            model_dict[model_name] = data[model_name]
            model_dict[model_name].loading_dir = info_flow[model_name]['model_dir']

        else:
            with open(info_flow[model_name]['model_dir'] + 'learning_params.yml', 'r') as ymlfile:
                cfg2 = yaml.safe_load(ymlfile)

            model_dict[model_name] = ref_model_dict[model_name].value(
                model_name=model_name,
                init_args=cfg2['info_flow'][model_name]['init_args'],
                device=device
            )
            model_dict[model_name].set_device(device)
            model_dict[model_name].load(
                epoch_num=info_flow[model_name]['epoch'],
                model_dir=info_flow[model_name]['model_dir']
            )
            model_dict[model_name].loading_dir = info_flow[model_name]['model_dir']
            model_dict[model_name].loading_epoch = info_flow[model_name]['epoch']
    print("\nFinished Initialization and Loading")
    return model_dict
