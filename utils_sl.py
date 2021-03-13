import random
import pickle
import numpy as np
import copy
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

import multinomial
import gaussian
import models_modules as mm
from collections import OrderedDict
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 

from typing import Tuple, Dict, List
from types import FunctionType

###############################################
##### Declaring possible loss functions
##############################################
def get_loss_and_eval_dict() -> Tuple[dict]:
    """ Returns a tuple with two elements, the first element is a dictionary
        containing a set of named loss functions, the second element is
        a dictionary containing a set of named evaluation metric functions
        These are generally useful loss and evaluation metric functions,
        but project specific losses can be added as well.
    """
    loss_dict = {
        'L2': LossWrapper(nn.MSELoss(reduction = "none"), "L2 or MSE"),
        'L1': LossWrapper(nn.L1Loss(reduction = "none"), "L1"),
        'L1Ensemble': LossWrapperEnsemble(nn.L1Loss(reduction = "none"), "L1 Ensemble"),
        'MultinomialNLL': LossWrapper(nn.CrossEntropyLoss(reduction = "none"), "Multinomial Negative Loglikelihood"),
        'BinomialNLL': LossWrapper(nn.BCEWithLogitsLoss(reduction = "none"), "Binomial Negative Loglikelihood"),
        'MultinomialEntropy': LossWrapper(multinomial.logits2ent, "MultinomialEntropy"),
        'MultinomialKL': LossWrapper(multinomial.logits2KL, "Multinomial KL Divergence"),
        'MultinomialKLEnsemble': LossWrapperEnsemble(multinomial.logits2KL, "Multinomial KL Divergence"),
        'GaussianNLL': LossWrapper(gaussian.negative_log_likelihood, "Gaussian Negative Loglikelihood"),
        'GaussianKL': LossWrapper(gaussian.divergence_KL, "Gaussian KL Divergence"),
        'IdentityLoss': LossWrapper(identity_function, "Identity Loss"),
    }

    eval_dict = {
        'MultinomialAccuracy': EvalMetricWrapper(multinomial.logits2acc, "Multinomial Accuracy"),
        'MultinomialAccuracyEnsemble': EvalMetricWrapperEnsemble(multinomial.logits2acc, "Multinomial Accuracy"),
        'MultinomialEntropy': EvalMetricWrapper(multinomial.logits2ent, "Multinomial Entropy"),
        'MultinomialEntropyEnsemble': EvalMetricWrapperEnsemble(multinomial.logits2ent, "Multinomial Entropy"),
        'ContinuousAccuracy': EvalMetricWrapper(continuous2accuracy, "Average Accuracy"),
        'ContinuousErrorMag': EvalMetricWrapper(continuous2error_mag, "Average Error Magnitude"),
        'ContinuousAccuracyEnsemble': EvalMetricWrapperEnsemble(continuous2accuracy, "Average Accuracy"),
        'ContinuousErrorMagEnsemble': EvalMetricWrapperEnsemble(continuous2error_mag, "Average Error Magnitude"),
        'BinomialAccuracy': EvalMetricWrapper(binomial_accuracy, "Binomial Accuracy")
    } 

    return loss_dict, eval_dict

def init_and_load_models(ref_model_dict : dict, 
                         info_flow : dict,
                         device : torch.device) -> Dict[str, mm.ModelWrapper]:
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
            model_dict[model_name] = ref_model_dict[model_name](model_name, info_flow[model_name]['init_args'], device = device)
            model_dict[model_name].set_device(device) 

        elif info_flow[model_name]['model_dir'][-3:] == 'pkl':
            data = torch.load(info_flow[model_name]['model_dir'])
            model_dict[model_name] = data[model_name]
            model_dict[model_name].loading_dir = info_flow[model_name]['model_dir']

        else:
            with open(info_flow[model_name]['model_dir'] + 'learning_params.yml', 'r') as ymlfile:
                cfg2 = yaml.safe_load(ymlfile)

            model_dict[model_name] = ref_model_dict[model_name](model_name, cfg2['info_flow'][model_name]['init_args'], device = device)
            model_dict[model_name].set_device(device) 

            model_dict[model_name].load(info_flow[model_name]['epoch'], info_flow[model_name]['model_dir'])
            model_dict[model_name].loading_dir = info_flow[model_name]['model_dir']
            model_dict[model_name].loading_epoch = info_flow[model_name]['epoch']

    print("\nFinished Initialization and Loading")
    return model_dict

class EvalMetricWrapper():
    """ A class which integrates logging the calculations of an
        evaluation function into the function

        Attributes:
            metric: a function which calculates a useful evaluation
            metric to understand how well the model is learning / performing

            metric_name: a string to use as the name of the metric when
            logging the results of the calculation
    """
    def __init__(self, metric_function : FunctionType, metric_name : str):
        """ Inits an instance of EvalMetricWrapper """
        self.metric = metric_function
        self.metric_name = metric_name

    def measure(self, input_tuple : tuple, logging_dict : dict, label : str):
        """ Calculates an evaluation metric based on the 'input_tuple' and
            stores the results in the 'logging_dict' using 'label' to
            generate its key
        """
        measurement = self.metric(*input_tuple)
        key = "{}/{}".format(label, self.metric_name)
        logging_dict['scalar'][key] = measurement.mean().item()

class LossWrapper():
    """ A class which integrates logging the calculations of an loss
        function into the function

        Attributes:
            loss_function: a function which calculates the loss of a
            batch of data points

            loss_name: a string to use as the name of the loss when
            logging the results of the calculation
    """
    def __init__(self, loss_function : FunctionType, loss_name : str):
        """ Inits an instance of LossWrapper """
        self.loss_function = loss_function
        self.loss_name = loss_name

    def forward(self,
                input_tuple : tuple,
                weight : float,
                logging_dict : dict,
                label : str) -> torch.Tensor:
        """ Calculates the loss based on the 'input_tuple', stores
            the results in the 'logging_dict' using 'label' to generate
            its key and then returns the loss torch.Tensor
        """
        loss = weight * self.loss_function(*input_tuple).mean()
        key = "{}/{}".format(label, self.loss_name)
        logging_dict['scalar'][key] = loss.item()
        return loss

class EvalMetricWrapperEnsemble(EvalMetricWrapper):
    """ A class which is basically the same as EvalMetricWrapper but
        targeted towards evaluating an ensemble of models.
    """
    def measure(self, input_tuple : tuple, logging_dict : dict, label : str):
        """ Calculates an evaluation metric based on the 'input_tuple' and
            stores the results in the 'logging_dict' using 'label' to
            generate its key for an ensemble of models
        """
        net_ests, target = input_tuple

        for i in range(net_ests.size(0)):
            net_est = net_ests[i]
            measurement = self.metric(net_est, target)
            key = "{}/{}_{}".format(label, self.metric_name, i)
            logging_dict['scalar'][key] = measurement.mean().item()

class LossWrapperEnsemble(LossWrapper):
    """ A class which is basically the same as LossWrapper but
        targeted towards evaluating an ensemble of models.
    """
    def forward(self,
                input_tuple : tuple,
                weight : float,
                logging_dict : dict,
                label : str) -> torch.Tensor:
        """ Calculates the loss based on the 'input_tuple', stores
            the results in the 'logging_dict' using 'label' to generate
            its key and then returns the loss torch.Tensor for an
            ensemble of models
        """
        net_ests, target = input_tuple
        loss = torch.zeros(1).float().to(net_ests.device)

        for i in range(net_ests.size(0)):
            net_est = net_ests[i]

            loss += weight * self.loss_function(net_est, target).mean()

        key = "{}/{}".format(label, self.loss_name)
        logging_dict['scalar'][key] = loss.item() / net_ests.size(0)

        return loss

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
        
def init_dataloader(cfg : dict, 
                    CustomDataset : torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
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
    idx_dict_path = cfg['dataloading_params']['idx_dict_path']

    #### loading previous val_train split to continue training a model
    if idx_dict_path is not None:
        with open(idx_dict_path, 'rb') as f:
            idx_dict = pickle.load(f)

        print("Loaded Train Val split dictionary from path: {}".format(idx_dict_path))
    else:
        idx_dict = None

    # Loading the dataset
    dataset = CustomDataset(cfg, idx_dict = idx_dict)

    sampler = SubsetRandomSampler(range(dataset.train_length))

    data_loader = DataLoader(dataset,
                             batch_size = batch_size,
                             num_workers = num_workers,
                             sampler = sampler,
                             pin_memory = True) 

    return data_loader

def save_as_yml(name : str, dictionary : dict, save_dir : str):
    """ Saves a dictionary as a yaml file """
    print("Saving ", name, " to: ", save_dir + name + ".yml")
    with open(save_dir + name + ".yml", 'w') as ymlfile2:
        yaml.dump(dictionary, ymlfile2)

def save_as_pkl(name : str, dictionary : dict, save_dir : str):
    """ Saves a dictionary as a pickle file """
    print("Saving ", name, " to: ", save_dir + name + ".pkl")
    with open(save_dir + name + '.pkl', 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)

# TODO: this function has not been debugged
# def calc_and_plot_tsne(points : np.ndarray, 
#                        labels : np.ndarray,
#                        log_dir : str):
#     """ Takes a set of high dimensional data points in a numpy array,
#         performs a tsne analysis on it (converting it to R2) and the 
#         plots a labelled graph of the results, saving the results as a png

#         See https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
#         for documentation on how to use TSNE
#     """   
#     tsne = TSNE(n_components=2,
#                 perplexity = 30.0,
#                 early_exaggeration = 12.0,
#                 learning_rate = 200.0,
#                 n_iter = 1000,
#                 method='barnes_hut')

#     print("Beginning TSNE")
#     Y = tsne.fit_transform(points)
#     print("Finished TSNE")
#     fig = plt.figure()
#     plt.scatter(Y[:,0], Y[:,1], c = labels)
#     plt.savefig("{}tsne_plot.png".format(log_dir))

def get_2Dconv_params(input_size : Tuple[int], 
                      output_size : Tuple[int]) -> List[Tuple[int]]:
    """ Calculates the parameters for each layer of a 2D convolutional
        neural network (CNN) given a tuple describing the input size of
        data points 'input_size' and a tuple describing the desired
        output size of data points 'output_size'
        
        NOTE: this function assumes input height and width have only 
        prime factors 2, 3, 5, 7 for now and the input height and width
        are divisible by their respective output height and width

        Returns:
            A list of tuples where each tuple in the list contains
            the parameters for a layer in the CNN
        
        Raises:
            ValueError: if input height or width are not divisble by
            their respective output height and width

    """
    input_chan, input_height, input_width = input_size
    output_chan, output_height, output_width = output_size

    chan_list = get_chan_list(input_chan, output_chan)

    if input_height % output_height == 0 and input_width % output_width == 0:
        prime_fact_rows = get_prime_fact(input_height // output_height)
        prime_fact_cols = get_prime_fact(input_width // output_width)
    else:
        raise ValueError("The input height and width are not divisible"
                        + " by the output height and width")
    # print(prime_fact_rows)
    # print(prime_fact_cols)

    if len(prime_fact_cols) > len(prime_fact_rows):

        while len(prime_fact_cols) > len(prime_fact_rows):
            prime_fact_rows.append(1)

    elif len(prime_fact_cols) < len(prime_fact_rows):

        while len(prime_fact_cols) < len(prime_fact_rows):
            prime_fact_cols.append(1)

    if len(prime_fact_cols) > len(chan_list):

        while len(prime_fact_cols) > len(chan_list):
            chan_list.append(chan_list[-1])

    elif len(prime_fact_cols) < len(chan_list):

        idx = 1

        while len(prime_fact_cols) < len(chan_list):

            prime_fact_cols.insert(idx, 1)
            prime_fact_rows.insert(idx, 1)

            idx += 2

            if idx >= len(prime_fact_cols):

                idx = 1

    e_p = np.zeros((8,len(prime_fact_rows))).astype(np.int16)

    chan_list.append(chan_list[-1])

    for idx in range(len(prime_fact_rows)):

        # first row input channel
        e_p[0, idx] = chan_list[idx]
        # second row output channel
        e_p[1, idx] = chan_list[idx + 1]
        # third row row kernel
        # fifth row row stride
        if prime_fact_rows[idx] == 7:
            e_p[2,idx] = 9
            e_p[4,idx] = 7
        elif prime_fact_rows[idx] == 5:
            e_p[2,idx] = 7
            e_p[4,idx] = 5
        elif prime_fact_rows[idx] == 3:
            e_p[2,idx] = 5
            e_p[4,idx] = 3
        elif prime_fact_rows[idx] == 2:
            e_p[2,idx] = 4
            e_p[4,idx] = 2
        else:
            e_p[2,idx] = 3
            e_p[4,idx] = 1
        # fourth row col kernel
        # sixth row col stride
        if prime_fact_cols[idx] == 7:
            e_p[3,idx] = 9
            e_p[5,idx] = 7
        elif prime_fact_cols[idx] == 5:
            e_p[3,idx] = 7
            e_p[5,idx] = 5
        elif prime_fact_cols[idx] == 3:
            e_p[3,idx] = 5
            e_p[5,idx] = 3
        elif prime_fact_cols[idx] == 2:
            e_p[3,idx] = 4
            e_p[5,idx] = 2
        else:
            e_p[3,idx] = 3
            e_p[5,idx] = 1

        # seventh row row padding
        e_p[6, idx] = 1
        # eighth row col padding
        e_p[7,idx] = 1

    e_p_list = []

    for idx in range(e_p.shape[1]):

        e_p_list.append(tuple(e_p[:,idx]))

    return e_p_list

def get_1Dconv_params(input_size : Tuple[int], 
                      output_size : Tuple[int]) -> List[Tuple[int]]:
    """ Calculates the parameters for each layer of a 1D convolutional
        neural network (1DCNN) given a tuple describing the input size of
        data points 'input_size' and a tuple describing the desired
        output size of data points 'output_size'
        
        NOTE: this function assumes input height and width have only 
        prime factors 2, 3, 5, 7 for now and the input width
        are divisible by its respective output width

        Returns:
            A list of tuples where each tuple in the list contains
            the parameters for a layer in the 1DCNN
        
        Raises:
            ValueError: if input height or width are not divisble by
            their respective output height and width

    """
    input_chan, input_width = input_size
    output_chan, output_width = output_size

    chan_list = get_chan_list(input_chan, output_chan)

    if input_width % output_width == 0:
        prime_fact= get_prime_fact(input_width // output_width)
    else:
        raise ValueError("The input  width is not divisible"
                        + " by the output width")

    if len(prime_fact) > len(chan_list):

        while len(prime_fact) > len(chan_list):
            chan_list.append(chan_list[-1])

    elif len(prime_fact) < len(chan_list):

        idx = 1

        while len(prime_fact) < len(chan_list):

            prime_fact.insert(idx, 1)

            idx += 2

            if idx >= len(prime_fact):

                idx = 1

    e_p = np.zeros((5,len(prime_fact))).astype(np.int16) #encoder parameters

    chan_list.append(chan_list[-1])

    for idx in range(len(prime_fact)):

        # print(prime_fact[idx])

        # first row input channel
        e_p[0, idx] = chan_list[idx]
        # second row output channel
        e_p[1, idx] = chan_list[idx + 1]
        # third row row kernel
        # fifth row row stride
        if prime_fact[idx] == 7:
            e_p[2,idx] = 9
            e_p[3, idx] = 7
        elif prime_fact[idx] == 5:
            e_p[2, idx] = 7
            e_p[3,idx] = 5
        elif prime_fact[idx] == 3:
            e_p[2,idx] = 5
            e_p[3, idx] = 3
        elif prime_fact[idx] == 2:
            e_p[2,idx] = 4
            e_p[3, idx] = 2
        else:
            e_p[2,idx] = 3
            e_p[3, idx] = 1
        # seventh row row padding
        e_p[4, idx] = 1

    e_p_list = []

    for idx in range(e_p.shape[1]):

        e_p_list.append(tuple(e_p[:,idx]))

    return e_p_list

def get_prime_fact(num : int) -> List[int]:
    """ Calculates a list of prime factors for an integer, but only
        supports integers which only have the prime factors 7, 5, 3, 2
        If a prime factor appears twice in the numbers factorization,
        it appears twice in the list that is returned.

        Raises:
            ValueError: if the integer has a prime factor other than
            7,5,3 or 2.
    """
    temp_num = num
    prime_fact_list = []

    while temp_num != 1:
        if temp_num % 7 == 0:
            temp_num = temp_num / 7
            prime_fact_list.append(7)
        elif temp_num % 5 == 0:
            temp_num = temp_num / 5
            prime_fact_list.append(5)
        elif temp_num % 3 == 0:
            temp_num = temp_num / 3
            prime_fact_list.append(3)
        elif temp_num % 2 == 0:
            temp_num = temp_num / 2
            prime_fact_list.append(2)
        else:
            raise ValueError("unsupported number " + str(num))

    if len(prime_fact_list) == 0:
        return []
    else:
        prime_fact_list.sort()
        prime_fact_list.reverse()

        return prime_fact_list 

def get_chan_list(input_size : int, output_size : int) -> List[int]:
    """ Calculates a list of channel sizes for a convolutional neural
        network's layers

        Args:
            input_size: an integer describing the network's first layer's
            channel size
            output_size: an integer describing the network's last layer's
            desired channel size
    """
    chan_list = [input_size]

    if input_size > output_size:
        chan_list = get_chan_list(output_size, input_size)
        chan_list.reverse()
        return chan_list

    elif input_size < output_size:
        if output_size % input_size == 0:
            prime_factors = get_prime_fact(output_size // input_size)
        else:
            chan_list.append(input_size)
            prime_factors = get_prime_fact(output_size)
            chan_list.append(prime_factors[0])
            prime_factors = prime_factors[1:]

        for factor in prime_factors:
            chan_list.append(chan_list[-1] * factor)

        return chan_list
    else:
        return [input_size, output_size]

def continuous2error_mag(ests : torch.Tensor,
                         gt_values : torch.Tensor) -> torch.Tensor:
    """ Calculates the magnitude of the error between two real vectors
        in an estimation problem.

        Args:
            ests: a torch.Tensor with an estimate of the ground truth 
            value from a model
            gt_values: a torch.Tensor with the the ground truth value of
            the vector being estimated

        Returns: 
            a torch.Tensor with the magnitude of the error between the
            vectors
        
        Raises
            ValueError: if the input tensors are not of a compatible rank
            ValueError: if the estimate and gt_values tensors are different
            sizes
    """
    num_size_dims = len(list(ests.size()))

    errors = gt_values - ests

    if num_size_dims <= 4:
        raise ValueError("this function only supports tensor's of rank 4 or less"
                        + "the input tensors are of rank {}".format(num_size_dims))

    elif ests.size(-1) == gt_values.size(-1):
        raise ValueError("the input tensors are not of the same size"
                         + "the estimate is of size {}".format(ests.size())
                         + "the ground truth values are of size {}".format(gt_values.size()))
    else:
        if num_size_dims == 1:
            errors_norm = torch.abs(errors)
            gt_values_norm = torch.abs(gt_values)
        elif num_size_dims == 2:
            errors_norm = errors.norm(p=2, dim =1)
            gt_values_norm = gt_values.norm(p=2, dim = 1)
        elif num_size_dims == 3:
            # print("error", errors.size())
            errors_norm = torch.sqrt(errors.pow(2).sum(1).sum(1))
            gt_values_norm = torch.sqrt(gt_values.pow(2).sum(1).sum(1))
        elif num_size_dims == 4:
            gt_values_norm = gt_values.norm(p=2, dim=3)
            errors_norm = errors.norm(p=2, dim=3)

        zero = torch.where(gt_values_norm != 0, 
                           torch.ones_like(errors_norm),
                           torch.zeros_like(errors_norm))
        
        return errors_norm * zero

def continuous2accuracy(ests : torch.Tensor,
                        gt_values : torch.Tensor) -> torch.Tensor:
    """ Calculates the 'accuracy' of an estimate in an estimation problem.

        Args:
            ests: a torch.Tensor with an estimate of the ground truth 
            value from a model
            labels: a torch.Tensor with the the ground truth value of
            the vector being estimated

        Returns: 
            a torch.Tensor with the 'accuracy' between the
            vectors
        
        Raises
            ValueError: if the input tensors are not of a compatible rank
            ValueError: if the estimate and labels tensors are different
            sizes
    """
    num_size_dims = len(list(ests.size()))

    errors = gt_values - ests

    if num_size_dims <= 4:
        raise ValueError("this function only supports tensor's of rank 4 or less"
                        + "the input tensors are of rank {}".format(num_size_dims))

    elif ests.size(-1) == gt_values.size(-1):
        raise ValueError("the input tensors are not of the same size"
                         + "the estimate is of size {}".format(ests.size())
                         + "the ground truth values are of size {}".format(gt_values.size()))
    else:
        if num_size_dims == 1:
            errors_norm = torch.abs(errors)
            gt_values_norm = torch.abs(gt_values)
        elif num_size_dims == 2:
            errors_norm = errors.norm(p=2, dim =1)
            gt_values_norm = gt_values.norm(p=2, dim = 1)
        elif num_size_dims == 3:
            # print("error", errors.size())
            errors_norm = torch.sqrt(errors.pow(2).sum(1).sum(1))
            gt_values_norm = torch.sqrt(gt_values.pow(2).sum(1).sum(1))
        elif num_size_dims == 4:
            gt_values_norm = gt_values.norm(p=2, dim=3)
            errors_norm = errors.norm(p=2, dim=3)

        zero = torch.where(gt_values_norm != 0, 
                           torch.ones_like(errors_norm), 
                           torch.zeros_like(errors_norm))
        
        accuracy = torch.where(errors_norm < gt_values_norm, 
                               errors_norm / gt_values_norm,
                               torch.ones_like(errors_norm))
        
        return accuracy * zero

def identity_function(ests : torch.Tensor) -> torch.Tensor:
    """Returns the input tensor"""
    return ests

def binomial_accuracy(logits : torch.Tensor,
                      labels : torch.Tensor) -> torch.Tensor:
    """ Calculates the accuracy of a binomial distributions parameters
        based on a single sample

        Args:
            inputs: a torch.Tensor with the parameters of binomial
            distributions
        Returns:
            a torch.Tensor of with accuracy for each
            binomial distribution
    """
    probs = torch.sigmoid(logits)
    samples = torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs))
    accuracy = torch.where(samples == labels, torch.ones_like(samples), torch.zeros_like(samples))

    return accuracy
