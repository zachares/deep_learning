import torch
import torch.nn as nn

from typing import List

##########################################
### Multi-layer Neural Network Modules ###
##########################################
class ModuleWrapper(nn.Module):
    """ a super class / wrapper for all multi-layer neural network
        modules with integrated methods for loading, saving and returning
        the parameters in the module.

        Attributes:
            model_name: a string with the model name of neural network
            module
            device: a torch.device that the module performs its
            calculations on
            model: the multi-layer neural network module which performs
            the calculations
    """
    def __init__(self, model_name : str, device : torch.device):
        """ Inits a ModuleWrapper instance """
        super().__init__()
        self.model_name = model_name
        self.device = device
        # self.parallel = False

    def set_device(self, device):
        """ sets the device of the model """
        self.device = device
        self.model = self.model.to(self.device)

    def forward(self, inputs : torch.Tensor) -> torch.Tensor:
        """ Returns the output of the module given a compatible input """
        return self.model(inputs)

    def weight_parameters(self) -> List[torch.Tensor]:
        """ Returns all the weight parameters in the module """
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self) -> List[torch.Tensor]:
        """ Returns all the bias parameters in the module. This is useful
            if you want to use a different weight initialization method
            for bias terms
        """
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def save(self, epoch_num : int, model_dir : str):
        """ Saves all the weights in the module to a directory """
        ckpt_path = '{}_{}'.format(model_dir + self.model_name, str(epoch_num).zfill(6))
        print("Saved Model to: ", ckpt_path)
        torch.save(self.model.state_dict(), ckpt_path)

    def load(self, epoch_num : int, model_dir : str):
        """ Loads all the weights in the module from a directory """
        ckpt_path = '{}_{}'.format(model_dir + self.model_name, str(epoch_num).zfill(6))
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt)
        print("Loaded Model to: ", ckpt_path)


###################################################
### Models Composed of Multiple Neural Networks ###
###################################################
class ModelWrapper(nn.Module):
    """ a super class / wrapper for all models composed of multiple
        neural network modules with integrated methods for loading,
        saving and returning the parameters in the model.

        Attributes:
            model_name: a string with the model name of neural network
            module
            device: a torch.device that the module performs its
            calculations on
            model: the multi-layer neural network module which performs
            the calculations
    """
    def __init__(self,
                 model_name : str,
                 device : torch.device = None,
                #  parallel : bool = False
                ):
        """ Inits a ModelWrapper Instance """
        super().__init__()
        self.model_name = model_name
        self.device = device
        # self.parallel = False

    def set_device(self, device):
        """ Sets the device of the modules within the model """
        self.device = device
        for model in self._modules.values():
            model.set_device(self.device)

    def save(self, epoch_num : int, model_dir : str):
        """ Saves all the neural network modules in the model to a
            directory
        """
        for model in self._modules.values():
            model.save(epoch_num, model_dir)

    def load(self, epoch_num : int, model_dir : str):
        """ Loads all the neural network modules in the model from a
            directory
        """
        for model in self._modules.values():
            model.load(epoch_num, model_dir)

    def set_uc(self, uc_bool : bool):
        """ Sets the instance attribute 'uc' in the neural network
            modules if they have that attribute.
        """
        for model in self._modules.values():
            if hasattr(model, 'uc'):
                model.uc = uc_bool
            elif hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)

    def parameters(self) -> List[torch.Tensor]:
        """ Returns the weights / parameters of all the modules in the
            model
        """
        parameters = []
        for model in self._modules.values():
            parameters += list(model.parameters())
        return parameters

    def train(self):
        """ Sets all the neural network modules in the model to training
            mode
        """
        for model in self._modules.values():
            model.train()

    def eval(self):
        """ Sets all the neural network modules in the model to
            evaluation mode
        """
        for model in self._modules.values():
            model.eval()
