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
from collections import OrderedDict
###############################################
##### Declaring possible loss functions
##############################################
def get_loss_and_eval_dict():
    loss_dict = {
        'MSE': Proto_Loss(nn.MSELoss(reduction = "none")),
        'L1': Proto_Loss(nn.L1Loss(reduction = "none")),
        'L1_Ensemble': Proto_Loss_Ensemble(nn.L1Loss(reduction = "none")),
        'Multinomial_NLL': Proto_Loss(nn.CrossEntropyLoss(reduction = "none")),
        'Multinomial_Margin_Loss': Proto_Loss(discrete_margin_loss),
        'Multinomial_NLL_Ensemble': Proto_Loss_Ensemble(nn.CrossEntropyLoss(reduction = "none")),
        # 'Weighted_Multinomial_NLL_Ensemble': Proto_Loss_Ensemble(nn.CrossEntropyLoss(weight = torch.tensor([1,1500]).float().to(self.device), reduction = "none")),
        'Binomial_NLL': Proto_Loss(nn.BCEWithLogitsLoss(reduction = "none")),
        'Multinomial_Entropy': Proto_Loss(multinomial.inputs2ent),
        'Multinomial_KL': Proto_Loss(multinomial.inputs2KL),
        'Multinomial_KL_Ensemble': Proto_Loss_Ensemble(multinomial.inputs2KL),
        'Gaussian_NLL': Proto_Loss(gaussian.negative_log_likelihood),
        'Gaussian_KL': Proto_Loss(gaussian.divergence_KL),
        'Identity_Loss': Proto_Loss(identity_loss),
    }

    eval_dict = {
        'Multinomial_Accuracy': Proto_Metric(multinomial.inputs2acc, ["accuracy"]),
        'Multinomial_Entropy': Proto_Metric(multinomial.inputs2ent_metric, ["entropy", "correxample_entropy", "incorrexample_entropy"]),
        'Multinomial_Accuracy_Ensemble': Proto_Metric_Ensemble(multinomial.inputs2acc, ["accuracy"]),
        'Multinomial_Entropy_Ensemble': Proto_Metric_Ensemble(multinomial.inputs2ent_metric, ["entropy", "correxample_entropy", "incorrexample_entropy"]),
        'Multinomial_Entropy_Spread_Ensemble': Proto_Metric_Ensemble(multinomial.inputs2gap, ["entropy_gap"]),
        'Gaussian_Error_Distrb': Proto_Metric(gaussian.params2error_metric, ["average_error", "covariance_error_Ratio"]),
        'Gaussian_Error_Samples': Proto_Metric(gaussian.samples2error_metric, ["average_error", "covariance_error_Ratio"]),
        'Continuous_Error': Proto_Metric(continuous2error_metric, ["average_accuracy", "average_error_mag"]),
        'Continuous_Error_Ensemble': Proto_Metric_Ensemble(continuous2error_metric, ["average_accuracy", "average_error_mag"]),
        'Sigmoid_Accuracy': Proto_Metric(sigmoid_accuracy_metric, ['binomial accuracy'])
    } 

    return loss_dict, eval_dict

def declare_models(ref_model_dict, cfg, device):
    info_flow = cfg['info_flow']
    model_dict = OrderedDict()
    ###############################################
    ##### Declaring models to be trained ##########
    #################################################
    for model_name in info_flow.keys():
        if model_name == "SAC_Policy":
            continue
        if info_flow[model_name]['model_folder'] == '':
            model_dict[model_name] = ref_model_dict[model_name](model_name, cfg['info_flow'][model_name]['init_args'], device = device)

        elif info_flow[model_name]['model_folder'][-3:] == 'pkl':
            data = torch.load(info_flow[model_name]['model_folder'])
            model_dict[model_name] = data[model_name]
            model_dict[model_name].loading_folder = info_flow[model_name]['model_folder']

        else:
            with open(info_flow[model_name]['model_folder'] + 'learning_params.yml', 'r') as ymlfile:
                cfg2 = yaml.safe_load(ymlfile)

            model_dict[model_name] = ref_model_dict[model_name](model_name, cfg2['info_flow'][model_name]['init_args'], device = device)

            model_dict[model_name].load(info_flow[model_name]['epoch'], info_flow[model_name]['model_folder'])
            model_dict[model_name].loading_folder = info_flow[model_name]['model_folder']
            model_dict[model_name].loading_epoch = info_flow[model_name]['epoch']       

    print("Finished Initialization")
    return model_dict

class Proto_Metric(object): 
    def __init__(self, metric_function, metric_names):
        self.metric = metric_function
        self.metric_names = tuple(metric_names)

    def measure(self, input_tuple, logging_dict, label, eval_dict):
        net_est, target = input_tuple

        measurements = self.metric(net_est, target)

        for i, metric_name in enumerate(self.metric_names):
            if type(measurements) == tuple:
                measurement = measurements[i]
            else:
                measurement = measurements

            logging_dict['scalar'][label + "/" + metric_name] = measurement.mean().item()

class Proto_Loss(object):
    def __init__(self, loss_function):
        self.loss_function = loss_function

    def forward(self, input_tuple, logging_dict, label, loss_dict):
        net_est, target = input_tuple

        if 'weight' in loss_dict.keys():
            weight = loss_dict['weight']
        else:
            weight = 1.0

        loss = weight * self.loss_function(net_est, target).mean()

        logging_dict['scalar'][label + "/loss"] = loss.item()

        return loss

class Proto_Metric_Ensemble(object):    
    def __init__(self, metric_function, metric_names):
        self.metric = metric_function
        self.metric_names = tuple(metric_names)

    def measure(self, input_tuple, logging_dict, label, args_tuple = None):
        net_ests, target = input_tuple

        for i in range(net_ests.size(0)):
            net_est = net_ests[i]

            measurements = self.metric(net_est, target)

            for j, metric_name in enumerate(self.metric_names):
                if type(measurements) == tuple:
                    measurement = measurements[j]
                else:
                    measurement = measurements

                logging_dict['scalar'][label + "/" + metric_name + "_" + str(i)] = measurement.mean().item()

class Proto_Loss_Ensemble(object):
    def __init__(self, loss_function):
        self.loss_function = loss_function

    def forward(self, input_tuple, logging_dict, weight, label, args_tuple = None):
        net_ests, target = input_tuple
        loss = torch.zeros(1).float().to(net_ests.device)

        for i in range(net_ests.size(0)):
            net_est = net_ests[i]

            loss += weight * self.loss_function(net_est, target).mean()

        logging_dict['scalar'][label + "/loss"] = loss.item() / net_ests.size(0)

        return loss

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, device = None):
        
        self.device = device

    def __call__(self, sample):

        # convert numpy arrays to pytorch tensors
        new_dict = dict()
        for k, v in sample.items():
        ###########################################################
        ##### Project Specific Code Here ##########################
        ###########################################################
            if k == "padding_mask":
                new_dict[k] = torch.from_numpy(v).bool()
            elif k[-3:] == "idx":
                new_dict[k] = torch.from_numpy(v).long()                
            else:
                new_dict[k] = torch.from_numpy(v).float()
            '''
            Default code is
            new_dict[k] = torch.from_numpy(v).float()
            '''
        ##########################################################    
        ##### End of Project Specific Code #######################
        ##########################################################

        return new_dict
        
def init_dataloader(cfg, Custom_Dataloader, device):
    ###############################################
    ########## Loading dataloader parameters ######
    ###############################################
    batch_size = cfg['dataloading_params']['batch_size']
    num_workers = cfg['dataloading_params']['num_workers']
    idx_dict_path = cfg['dataloading_params']['idx_dict_path']
    run_mode = cfg['training_params']['run_mode'] 
    val_ratio = cfg['training_params']['val_ratio']

    #### loading previous val_train split to continue training a model
    if idx_dict_path is not None:
        with open(idx_dict_path, 'rb') as f:
            idx_dict = pickle.load(f)

        print("Loaded Train Val split dictionary from path: " + idx_dict_path)
    else:
        idx_dict = None

    dataset = Custom_Dataloader(cfg, idx_dict = idx_dict, device = device, transform=transforms.Compose([ToTensor(device = device)]))

    if val_ratio == 0:
        print("No validation set")

    if run_mode == 0:
        train_sampler = SubsetRandomSampler(range(dataset.dev_length))
        dataset.dev_bool = True
    else:
        train_sampler = SubsetRandomSampler(range(dataset.train_length))

    data_loader = DataLoader(dataset, batch_size= batch_size, num_workers=num_workers, sampler=train_sampler, pin_memory = True) 

    val_data_loader = None  

    if val_ratio != 0:
        val_sampler = SubsetRandomSampler(range(dataset.val_length))
        val_dataset = copy.deepcopy(dataset)
        val_dataset.val_bool = True
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, sampler= val_sampler, pin_memory = True)

    return data_loader, val_data_loader

def get_2Dconv_params(input_size, output_size):

    input_chan, input_height, input_width = input_size
    output_chan, output_height, output_width = output_size
    # assume input height and width have only two prime factors 2, 3, 5, 7 for now

    chan_list = get_chan_list(input_chan, output_chan)

    if input_height % output_height == 0 and input_width % output_width == 0:
        prime_fact_rows = get_prime_fact(input_height // output_height)
        prime_fact_cols = get_prime_fact(input_width // output_width)
    else:
        raise Exception("these input output image sizes are not supported" + str(num_size_dims))
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

    e_p = np.zeros((8,len(prime_fact_rows))).astype(np.int16) #encoder parameters

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

def get_1Dconv_params(input_size, output_size):

    input_chan, input_width = input_size
    output_chan, output_width = output_size

    # assume input height have only two prime factors 2, 3, 5 and 7 for now

    # assume input height and width have only two prime factors 2, 3, 5, 7 for now
    chan_list = get_chan_list(input_chan, output_chan)

    if input_width % output_width == 0:
        prime_fact= get_prime_fact(input_width // output_width)
    else:
        raise Exception("these input output image sizes are not supported" + str(num_size_dims))

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

def get_prime_fact(num):

    #assume num is factorized by powers of 2 and 3
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
            raise Exception("unsupported number " + str(num))

    if len(prime_fact_list) == 0:
        return []
    else:
        prime_fact_list.sort()
        prime_fact_list.reverse()

        return prime_fact_list 

def get_chan_list(input_size, output_size):

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

def continuous2error_metric(preds, labels):
    num_size_dims = len(list(preds.size()))

    # print("pred", preds.size())
    # print("label", labels.size())

    errors = labels - preds

    if num_size_dims <= 4:
        if num_size_dims == 1:
            errors_norm = torch.abs(errors)
            labels_norm = torch.abs(labels)
        elif num_size_dims == 2:
            errors_norm = errors.norm(p=2, dim =1)
            labels_norm = labels.norm(p=2, dim = 1)
        elif num_size_dims == 3:
            # print("error", errors.size())
            errors_norm = torch.sqrt(errors.pow(2).sum(1).sum(1))
            labels_norm = torch.sqrt(labels.pow(2).sum(1).sum(1))
        elif num_size_dims == 4:
            labels_norm = labels.norm(p=2, dim=3)
            errors_norm = errors.norm(p=2, dim=3)

        zero = torch.where(labels_norm != 0, torch.ones_like(errors_norm), torch.zeros_like(errors_norm))

        accuracy = torch.where(errors_norm < labels_norm, errors_norm / labels_norm, torch.ones_like(errors_norm))
        
        return accuracy * zero, errors_norm * zero
    else:
        raise Exception("estimates tensor size invalid with number of dimensions" + str(num_size_dims))

def identity_loss(preds, labels):
    return preds

def discrete_margin_loss(logprobs, labels):
    probs = F.softmax(logprobs, dim = 1)
    goal_margin = 0.02

    probs[torch.arange(probs.size(0)), labels] = probs[torch.arange(probs.size(0)), labels] - goal_margin

    prob_diff = probs - probs[torch.arange(probs.size(0)), labels].unsqueeze(1).repeat_interleave(probs.size(1), 1)

    margin_test = torch.where(prob_diff > 0, torch.ones_like(prob_diff), torch.zeros_like(prob_diff)).sum(1)
    
    return -logprobs[torch.arange(probs.size(0)), labels] * torch.where(margin_test > 0, torch.ones_like(margin_test), torch.zeros_like(margin_test))


def sigmoid_accuracy_metric(preds, labels):
    probs = torch.sigmoid(preds)
    samples = torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs))
    accuracy = torch.where(samples == labels, torch.ones_like(samples), torch.zeros_like(samples))

    return accuracy

# def histogram_loss(est, target, loss_function, hyperparameter):
#     errors = loss_function(net_est, target).sum(1)   

#     mu_err = errors.mean()
#     std_err = errors.std()

#     if 1 - torch.sigmoid(torch.log(self.hyperparameter * std_err / mu_err)) < 0.1:
#         num_batch = np.round(0.1 * errors.size(0)).astype(np.int32)
#     else:
#         num_batch = torch.round((1 - torch.sigmoid(torch.log( self.hyperparameter * std_err / mu_err))) * errors.size(0)).type(torch.int32)

#     errors_sorted, indices = torch.sort(errors)

#     return errors_sorted[-num_batch:]

# def pointwise_distance_loss(z, p):

#     z_dist = (z.unsqueeze(2).repeat(1,1,z.size(0)) - z.unsqueeze(2).repeat(1,1,z.size(0)).transpose(0,2)).norm(p=2, dim = 1)

#     p_dist = (p.unsqueeze(2).repeat(1,1,p.size(0)) - p.unsqueeze(2).repeat(1,1,p.size(0)).transpose(0,2)).norm(p=2, dim = 1)

#     element_wise = (z_dist - p_dist).norm(p=2, dim = 1)

#     return element_wise

# def ranking_loss(values, targets):

#     idxs = torch.argsort(targets)
#     v_sorted = values[idxs] - values.min()

#     v_mat = v_sorted.unsqueeze(0).repeat_interleave(v_sorted.size(0), dim = 0)

#     v_mat_diag = v_mat.diag().unsqueeze(1).repeat_interleave(v_sorted.size(0), dim = 1)
    
#     v = (v_mat - v_mat_diag)

#     w_bad = torch.where(v <= 0, torch.ones_like(v), torch.zeros_like(v)).triu()
#     w_bad[torch.arange(w_bad.size(0)), torch.arange(w_bad.size(1))] *= 0

#     w_good = (torch.where(v > 0, torch.ones_like(v), torch.zeros_like(v)) * torch.where(v <= 1, torch.ones_like(v), torch.zeros_like(v))).triu()
#     w_good[torch.arange(w_good.size(0)), torch.arange(w_good.size(1))] *= 0

#     return (w_good * v + w_bad * v)


# def simple_normalization(vector, dim = 1):
#     return vector / vector.norm(p=2, dim = dim).unsqueeze(dim).repeat_interleave(vector.size(dim), dim = dim)