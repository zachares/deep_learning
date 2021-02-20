import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

# NOTE: these functions do not support binomial distributions 
# represented by a single parameter

def logits2inputs(logits : torch.Tensor) -> torch.Tensor:
    """ Calculates probabilities of a multinomial distribution from a
        vector of logits using the softmax function

        NOTE: the input logits should be of size NxM, where N is the
        size of the batch of data points being processed and M is the 
        number of categories that the multinomial's random variable can
        take on

        Args:
            logits: a torch.Tensor with real values
        
        Returns:
            a torch.Tensor of size 2xNxM (explained above) with the 
            probabilities of the multinomial distributions being the first
            row and the logprobabilities of the multinomial 
            distribution being in the second row
        
        Raises:
            ValueError: if the input tensor is not of rank 2
    """
    num_size_dims = len(list(logits.size()))
      
    if num_size_dims == 2:
        probs = F.softmax(logits, dim = 1)
        logprobs = F.log_softmax(logits, dim = 1)
    else:
        raise ValueError("estimates tensor size invalid with number"
                         + " of dimensions {}".format(num_size_dims))

    return torch.cat([probs.unsqueeze(0), logprobs.unsqueeze(0)], dim = 0)

def probs2inputs(probs : torch.Tensor) :
    """ Args:
            probs: a torch.Tensor with of size NxM (N = batch size,
            M = categories the random variable can take on) of 
            multinomial probabilities
        
        Returns:
            a torch.Tensor of size 2xNxM (explained above) with the 
            probabilities of the multinomial distributions being the first
            row and the logprobabilities of the multinomial 
            distribution being in the second row
        
        Raises:
            ValueError: if the input tensor is not of rank 2
    """
    num_size_dims = len(list(probs.size()))

    if num_size_dims == 2:
        logprobs = torch.log(probs)
    else:
        raise ValueError("estimates tensor size invalid with number"
                         + " of dimensions {}".format(num_size_dims))

    return torch.cat([probs.unsqueeze(0), logprobs.unsqueeze(0)], dim = 0)

def inputs2KL(inputs0 : torch.Tensor, inputs1 : torch.Tensor) -> torch.Tensor:
    """ Calculates the KL divergence between two multinomial distributions

        Args:
            inputs0: a torch.Tensor of size NxM (N = batch size,
            M = number of categories) with the parameters of multinomial
            distributions
            inputs1: a torch.Tensor of size NxM (N = batch size,
            M = number of categories) with the parameters of multinomial
            distributions
        
        Returns:
            a torch.Tensor of size N with KL divergence value for each
            multinomial distribution pair
    """
    probs0, logprobs0 = inputs0[0], inputs0[1]
    probs1, logprobs1 = inputs1[0], inputs1[1]

    nonzero0 = torch.where(probs0 != 0, torch.ones_like(probs0), torch.zeros_like(probs0))
    nonzero1 = torch.where(probs1 != 0, torch.ones_like(probs1), torch.zeros_like(probs1))

    return torch.where((nonzero0 * nonzero1) != 0, 
                        -probs1 * (logprobs0 - logprobs1), 
                        torch.zeros_like(probs1)).sum(-1)

def inputs2ent(inputs : torch.Tensor) -> torch.Tensor:
    """ Calculates the entropy of a multinomial distribution

        Args:
            inputs: a torch.Tensor of size NxM (N = batch size,
            M = number of categories) with the parameters of multinomial
            distributions
        Returns:
            a torch.Tensor of size N with entropy for each
            multinomial distribution
    """
    probs, logprobs = inputs[0], inputs[1]
    return torch.where(probs != 0,
                       -1.0 * probs * logprobs,
                       torch.zeros_like(probs)).sum(-1)

def inputs2acc(inputs, samples):
    """ Calculates the accuracy of a multinomial distributions parameters
        based on a single sample

        Args:
            inputs: a torch.Tensor of size NxM (N = batch size,
            M = number of categories) with the parameters of multinomial
            distributions
        Returns:
            a torch.Tensor of size N with accuracy for each
            multinomial distribution
    """
    probs, logprobs = inputs[0], inputs[1]
    est_idx = probs.max(1)[1]

    return torch.where(samples == est_idx,
                       torch.ones_like(samples),
                       torch.zeros_like(samples)).float()

# def inputs2gap(inputs, labels):
#     entropy = inputs2ent(inputs)
#     entropy_sorted, indices = torch.sort(entropy, dim = 0)

#     ent_diff = entropy_sorted[1:] - entropy_sorted[:-1]

#     return torch.max(ent_diff) - torch.min(ent_diff)