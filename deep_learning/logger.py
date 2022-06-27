from collections  import defaultdict
import numpy as np
import wandb

class Logger():
    """ a class which logs scalars to weights and biases

        Attributes:
            logging_dir: a string of the path to the directory where the
            results will be saved

            mean_dict: a dictionary where all the scalar metrics of training
            are stored such that the average metric over an entire epoch
            of training / data set can be calculated
    """
    def __init__(self, logging_bool : bool = False):
        """ Inits a Logger Instance """
        self.logging_bool  = logging_bool
        self.logging_dict = {}
        self.logging_dict['scalar'] = {}
        self.mean_dict = {}
        self.mean_dict['scalar'] = defaultdict(list)

    def log_scalars(self):
        """ Saves all scalar training results currently in the attribute
            logging_dict and stores the current value for each scalar in
            a list in a seperate dictionary
        """
        for key in self.logging_dict['scalar'].keys():
            scalar = self.logging_dict['scalar'][key]
            if self.logging_bool:
                wandb.log({f"{self.label}_{key}": scalar})
            self.mean_dict['scalar'][key].append(scalar)

    def get_mean_dict(self):
        return {
            key: np.round(np.mean(value), 5)
            for key, value in self.mean_dict['scalar'].items()
        }

    def log_means(self):
        """ Saves the mean value of all scalar metrics that are stored
            in the mean_dict. See log_scalars method to see how scalars
            are stored in mean_dict.
        """
        for key in self.mean_dict['scalar'].keys():
            scalar = np.mean(self.mean_dict['scalar'][key])
            if self.logging_bool:
                wandb.log({f"{self.label}_{key}_mean": scalar})
        self.mean_dict['scalar'].clear()
