import numpy as np
import torch
from tensorboardX import SummaryWriter
from typing import Dict

class Logger():
    """ a class which setups a directory for logging neural network
        training results and saves results to a file this directory

        Attributes:
            logging_dir: a string of the path to the directory where the
            results will be saved
            writer: a SummaryWriter object which saves values to a file
            logging_dict: a dictionary where the training results for
            each iteration are stored until they are saved in the
            SummaryWriter
    """
    def __init__(self, logging_dir : str):
        """ Inits a Logger Instance """
        self.logging_dir = logging_dir
        self.writer = SummaryWriter(self.logging_dir)

        self.logging_dict = {}
        self.logging_dict['scalar'] = {}
        self.logging_dict['image'] = {}
    
    def log_results(self,
                    iteration : int,
                    label : str,
                    save_image : bool = False):
        """ saves the current results in logging_dict, the results
            can be a scalar such as the value of a loss function or
            a 2D image

            Args:
                iteration : an integer indicating the time in the training run
                label: a string describing whether the result was during
                training or evaluation
                save_image: a boolean indicating whether to save the images
                from the current results of training
        
        """
        self.iteration = iteration
        self.label = label

        if len(self.logging_dict['scalar'].keys()) != 0:
            self.log_scalars()

        if len(list(self.logging_dict['image'].keys())) != 0 and save_image:
            self.log_images2D()

    def log_scalars(self):
        """ Saves all scalar training results currently in the attribute 
            logging_dict
        """
        for key in self.logging_dict['scalar'].keys():
            # print(key, logging_dict['scalar'][key])
            self.writer.add_scalar(self.label + key, 
                                   self.logging_dict['scalar'][key], 
                                   self.iteration)

    def log_images2D(self):
        """ Saves all 2D images from training results currently in the 
            attribute logging_dict 
        """
        for image_key in self.logging_dict['image']:
            image_list = self.logging_dict['image'][image_key]
            if len(image_list) != 0:
                for idx, image in enumerate(image_list):
                    image_list[idx] = image.detach().cpu().numpy()
                    image_array = np.rot90(np.concatenate(image_list, axis = 1),
                                           k = 3, 
                                           axes=(1,2)).astype(np.uint8)

                    self.writer.add_image("{}{}predicted_image".format(self.label,image_key),
                                          image_array,
                                          self.iteration)
