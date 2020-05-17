import numpy as np
import torch
import torch.nn.functional as F
import copy
import h5py

def read_h5(path):
    return h5py.File(path, 'r', swmr=True, libver = 'latest')

def continuous2error_metric(preds, labels):
    num_size_dims = len(list(preds.size()))

    errors = labels - preds

    if num_size_dims == 1 or num_size_dims == 2:
        if num_size_dims == 1:
            errors_norm = torch.abs(errors)
            labels_norm = torch.abs(labels)
        else:
            errors_norm = errors.norm(p=2, dim =1)
            labels_norm = labels.norm(p=2, dim = 1)

        accuracy = torch.where(torch.abs(errors_norm - labels_norm) < labels_norm, errors_norm / labels_norm, torch.zeros_like(errors_norm))
        return accuracy, errors_norm
    else:
        raise Exception("estimates tensor size invalid with number of dimensions" + str(num_size_dims))

def histogram_loss(est, target, loss_function, hyperparameter):
    errors = loss_function(net_est, target).sum(1)   

    mu_err = errors.mean()
    std_err = errors.std()

    if 1 - torch.sigmoid(torch.log(self.hyperparameter * std_err / mu_err)) < 0.1:
        num_batch = np.round(0.1 * errors.size(0)).astype(np.int32)
    else:
        num_batch = torch.round((1 - torch.sigmoid(torch.log( self.hyperparameter * std_err / mu_err))) * errors.size(0)).type(torch.int32)

    errors_sorted, indices = torch.sort(errors)

    return errors_sorted[-num_batch:]

def pointwise_distance_loss(z, p):

    z_dist = (z.unsqueeze(2).repeat(1,1,z.size(0)) - z.unsqueeze(2).repeat(1,1,z.size(0)).transpose(0,2)).norm(p=2, dim = 1)

    p_dist = (p.unsqueeze(2).repeat(1,1,p.size(0)) - p.unsqueeze(2).repeat(1,1,p.size(0)).transpose(0,2)).norm(p=2, dim = 1)

    element_wise = (z_dist - p_dist).norm(p=2, dim = 1)

    return element_wise

def ranking_loss(values, targets):

    idxs = torch.argsort(targets)
    v_sorted = values[idxs] - values.min()

    v_mat = v_sorted.unsqueeze(0).repeat_interleave(v_sorted.size(0), dim = 0)

    v_mat_diag = v_mat.diag().unsqueeze(1).repeat_interleave(v_sorted.size(0), dim = 1)
    
    v = (v_mat - v_mat_diag)

    w_bad = torch.where(v <= 0, torch.ones_like(v), torch.zeros_like(v)).triu()
    w_bad[torch.arange(w_bad.size(0)), torch.arange(w_bad.size(1))] *= 0

    w_good = (torch.where(v > 0, torch.ones_like(v), torch.zeros_like(v)) * torch.where(v <= 1, torch.ones_like(v), torch.zeros_like(v))).triu()
    w_good[torch.arange(w_good.size(0)), torch.arange(w_good.size(1))] *= 0

    return (w_good * v + w_bad * v)


def simple_normalization(vector, dim = 1):
    return vector / vector.norm(p=2, dim = dim).unsqueeze(dim).repeat_interleave(vector.size(dim), dim = dim)

def get_2Dconv_params(input_size, output_size):

    input_chan, input_height, input_width = input_size
    output_chan, output_height, output_width = output_size

    # assume input height and width have only two prime factors 2, 3, 5, 7 for now

    if output_chan % input_chan == 0:
        chan_factors_list = [input_chan]
    elif output_chan % 8 == 0:
        chan_factors_list = [input_chan, 8]
    else:
        raise Exception("these input output channel sizes are not supported" + str(num_size_dims))

    chan_factors_list = chan_factors_list + get_prime_fact(output_chan // input_chan)

    chan_list = []

    for i, factor in enumerate(chan_factors_list):
        if i == 0:
            chan_list.append(factor)
        else:
            chan_list.append(chan_list[-1] * factor)

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

    if output_chan % input_chan == 0:
        chan_factors_list = [input_chan]
    elif output_chan % 8 == 0:
        chan_factors_list = [input_chan, 8]
    else:
        raise Exception("these input output channel sizes are not supported" + str(num_size_dims))

    chan_factors_list = chan_factors_list + get_prime_fact(output_chan // input_chan)

    chan_list = []

    for i, factor in enumerate(chan_factors_list):
        if i == 0:
            chan_list.append(factor)
        else:
            chan_list.append(chan_list[-1] * factor)

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
    temp_num = copy.copy(num)
    prime_fact_list = []

    while temp_num != 1:

        if temp_num % 7 == 0:
            temp_num = temp_num / 7
            prime_fact_list.append(7)
        elif temp_num % 5 == 0:
            temp_num = temp_num / 5
            prime_fact_list.append(5)
        if temp_num % 3 == 0:
            temp_num = temp_num / 3
            prime_fact_list.append(3)
        elif temp_num % 2 == 0:
            temp_num = temp_num / 2
            prime_fact_list.append(2)

    prime_fact_list.sort()
    prime_fact_list.reverse()

    return prime_fact_list 

