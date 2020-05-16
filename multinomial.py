import numpy as np
import torch
import torch.nn.functional as F

def logits2KL(logits0, logits1):
    num_size_dims = len(list(logits0.size()))

    if num_size_dims == 1:
        probs0 = torch.sigmoid(logits_0)
        probs1 = torch.sigmoid(logits_1)
        return probs0 * (torch.log(probs0) - torch.log(probs1)) + (1 - probs1) * (torch.log(1 - probs0) - torch.log(1 - probs1))
    elif num_size_dims == 2:
        return -(F.softmax(logits1, dim =1) * (F.log_softmax(logits0, dim = 1) - F.log_softmax(logits1, dim = 1))).sum(1)
    else:
        raise Exception("estimates tensor size invalid with number of dimensions" + str(num_size_dims))

def probs2KL(probs0, probs1):
    num_size_dims = len(list(probs0.size()))

    if num_size_dims == 1:
        return probs0 * (torch.log(probs0) - torch.log(probs1)) + (1 - probs1) * (torch.log(1 - probs0) - torch.log(1 - probs1))
    elif num_size_dims == 2:
        return -(probs1 * (torch.log(probs0) - torch.log(probs1))).sum(1)
    else:
        raise Exception("estimates tensor size invalid with number of dimensions" + str(num_size_dims))

def logits2probs(logits):
    num_size_dims = len(list(logits.size()))

    if  num_size_dims == 1:
        probs = torch.sigmoid(logits)
    elif num_size_dims == 2:
        probs = F.softmax(logits, dim = 1)
    else:
        raise Exception("estimates tensor size invalid with number of dimensions" + str(num_size_dims))

    return probs

def probs2ent(probs): # from a binomial or multinomial distribution
    num_size_dims = len(list(probs.size()))

    if  num_size_dims == 1:
        return  -torch.log(probs) * probs - torch.log(1-probs) * (1 - probs)
    elif num_size_dims == 2:
        return torch.where(probs != 0, -1.0 * probs * torch.log(probs), torch.zeros_like(probs)).sum(-1)
    else:
        raise Exception("estimates tensor size invalid with number of dimensions" + str(num_size_dims))

def logits2ent(logits): # from a binomial or multinomial distribution
    num_size_dims = len(list(logits.size()))

    if  num_size_dims == 1:
        probs = torch.sigmoid(logits)
        return  -torch.log(probs) * probs - torch.log(1-probs) * (1 - probs)
    elif num_size_dims == 2:
        probs = F.softmax(logits, dim = 1)
        return torch.where(probs != 0, -1.0 * probs * F.log_softmax(logits, dim = 1), torch.zeros_like(probs)).sum(-1)
    else:
        raise Exception("estimates tensor size invalid with number of dimensions" + str(num_size_dims))

def logits2acc(logits, label):
    num_size_dims = len(list(logits.size()))

    if  num_size_dims == 1:
        probs = torch.sigmoid(logits)
        est_idx = torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs))
    elif num_size_dims == 2:
        probs = F.softmax(logits, dim = 1)
        est_idx = probs.max(1)[1]
    else:
        raise Exception("estimates tensor size invalid with number of dimensions" + str(num_size_dims))

    return torch.where(label == est_idx, torch.ones_like(label), torch.zeros_like(label)).float()

def probs2acc(probs, label):
    num_size_dims = len(list(probs.size()))

    if  num_size_dims == 1:
        est_idx = torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs))
        label_idx = label.clone()
    elif num_size_dims == 2:
        est_idx = est.max(1)[1]
    else:
        raise Exception("estimates tensor size invalid with number of dimensions" + str(num_size_dims))

    return torch.where(label == est_idx, torch.ones_like(label), torch.zeros_like(label))

def logits2ent_metric(logits, label):
    accuracy = logits2acc(logits, label)
    entropy = logits2ent(logits)

    positive_entropy = entropy[accuracy.nonzero()]
    negative_entropy = entropy[(1 - accuracy).nonzero()]

    return entropy, positive_entropy, negative_entropy

