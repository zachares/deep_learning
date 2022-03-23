import enum
import torch
import numpy as np
from typing import Dict

from deep_learning.models_manager.model_wrappers import ModelWrapper
from deep_learning.models_manager.cnns import CONV2DN
from deep_learning.models_manager.mlps import ResNetFCN


class Cifar10Classifier(ModelWrapper):
    """ A class whose model (based on neural networks) classifies the
        object or animal in a 32x32 RGB image amongst 10 categories

        Attributes:
            image_size: a list of integers describing the size of an
            input image

            encoding_size: a list of integers describing the desired size
            of the output of the image processing CNN

            num_resnet_layers: the number of resnet layers at the end of
            the network

            device: a reference to the hardware device (GPU or CPU) that
            an instance of the class will perform its calculations on

            model_name: a string naming the model for saving and loading
            trained models
    """
    def __init__(self,
                 model_name : str,
                 init_args : dict,
                 device : torch.device = None):
        """ Inits an instance of the Cifar10Classifier class"""
        super().__init__(model_name, device)

        self.image_size = init_args['image_size']
        self.encoding_size = init_args['encoding_size']
        self.num_resnet_layers = init_args['num_resnet_layers']

        self.image_processor = CONV2DN(
            model_name=model_name,
            input_size=self.image_size,
            output_size=self.encoding_size,
            nonlinear=True,
            batchnorm=True,
            dropout=False,
            device=self.device
        ).to(self.device)

        self.encoding_classifier = ResNetFCN(
            model_name=model_name,
            input_size=self.encoding_size[0],
            hidden_size=self.encoding_size[0],
            output_size=10,
            num_layers=self.num_resnet_layers,
            device=self.device
        ).to(self.device)

    def forward(
        self,
        input_dict : Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
            Args:
                input_dict: a dictionary with values of torch.Tensors
                and keys composed of strings containing the inputs to the
                model

            Returns:
                a dictionary of torch.Tensors which are the outputs of
                the model and can be used to classify the class of an
                input image
        """
        encs = torch.flatten(
            self.image_processor(input_dict['image'].transpose(1,3)),
            start_dim=1
        )
        class_logits = self.encoding_classifier(encs)
        return {
            'class_logits' : class_logits,
        }

    def classify(self, images_np : np.ndarray) -> torch.Tensor:
        """ Classifies an batch of images stored in np.ndarrays and
            returns the logits outputted by the instance's model

            Note: this method assumes that images come in batches
            i.e. they are of size Nx32x32x3 where N is the batch size
            if you want to classify a single image, an extra dimension
            needs to be added onto the input image to make it of shape
            1x32x32x3 before it is inputted as an argument to this method

            Args:
                images_np: a np.ndarray with a batch of images for the
                model to classify

            Returns:
                a torch.Tensor of the classification logits for each batch
        """
        with torch.no_grad():
            self.eval()
            output_dict = self.forward({
                'image' : torch.from_numpy(images_np).to(self.device).float()
            })
        return output_dict['class_logits']

class ModelFactory(enum.Enum):
    Cifar10Classifier = Cifar10Classifier
