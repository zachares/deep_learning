import numpy as np
import torch
import torch.nn.functional as F

### Does not support binomial distributions represented by a single parameter

EPS = 1e-6

def logits2dirprobs(logits):
    weights = F.relu(logits) + 1.0

    probs = weights / weights.sum(1).unsqueeze(1).repeat_interleave(weights.size(1), dim = 1)

    # print(probs)

    return probs

def logits2inputs(logits):
    num_size_dims = len(list(logits.size()))

    if num_size_dims == 2:
        probs = F.softmax(logits, dim = 1)
        logprobs = F.log_softmax(logits, dim = 1)
    else:
        raise Exception("estimates tensor size invalid with number of dimensions " + str(num_size_dims))

    return torch.cat([probs.unsqueeze(0), logprobs.unsqueeze(0)], dim = 0)

def probs2inputs(probs):
    num_size_dims = len(list(probs.size()))

    if num_size_dims == 2:
        logprobs = torch.log(probs)
    else:
        raise Exception("estimates tensor size invalid with number of dimensions " + str(num_size_dims))

    return torch.cat([probs.unsqueeze(0), logprobs.unsqueeze(0)], dim = 0)

def logits2probs(logits):
    num_size_dims = len(list(logits.size()))

    if num_size_dims == 2:
        probs = F.softmax(logits, dim = 1)
        probs = torch.where(probs !=0, probs, EPS * torch.ones_like(probs))
        probs = probs / probs.sum(1).unsqueeze(1).repeat_interleave(probs.size(1), dim = 1)
        return probs
    else:
        raise Exception("estimates tensor size invalid with number of dimensions " + str(num_size_dims))

def inputs2KL(inputs0, inputs1):
    probs0, logprobs0 = inputs0[0], inputs0[1]
    probs1, logprobs1 = inputs1[0], inputs1[1]

    nonzero0 = torch.where(probs0 != 0, torch.ones_like(probs0), torch.zeros_like(probs0))
    nonzero1 = torch.where(probs1 != 0, torch.ones_like(probs1), torch.zeros_like(probs1))

    return torch.where((nonzero0 * nonzero1) != 0, -probs1 * (logprobs0 - logprobs1), torch.zeros_like(probs1)).sum(-1)

def inputs2ent(inputs): # from a multinomial distribution
    probs, logprobs = inputs[0], inputs[1]
    return torch.where(probs != 0, -1.0 * probs * torch.log(probs), torch.zeros_like(probs)).sum(-1)

def inputs2acc(inputs, labels):
    probs, logprobs = inputs[0], inputs[1]
    est_idx = probs.max(1)[1]

    return torch.where(labels == est_idx, torch.ones_like(labels), torch.zeros_like(labels)).float()

def inputs2gap(inputs, labels):
    entropy = inputs2ent(inputs)
    entropy_sorted, indices = torch.sort(entropy, dim = 0)

    ent_diff = entropy_sorted[1:] - entropy_sorted[:-1]

    return torch.max(ent_diff) - torch.min(ent_diff)
    
def inputs2ent_metric(inputs, labels):
    accuracy = inputs2acc(inputs, labels)
    entropy = inputs2ent(inputs)

    positive_entropy = entropy[accuracy.nonzero()]
    negative_entropy = entropy[(1 - accuracy).nonzero()]

    return entropy, positive_entropy, negative_entropy

