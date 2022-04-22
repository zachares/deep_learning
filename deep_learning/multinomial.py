import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

# NOTE: these functions do not support binomial distributions
# represented by a single parameter

def logits2inputs(logits : torch.Tensor) -> Tuple[torch.Tensor]:
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


def probs2inputs(probs : torch.Tensor) -> Tuple[torch.Tensor]:
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

    if num_size_dims <= 2:
        logprobs = torch.log(probs)
    else:
        raise ValueError("estimates tensor size invalid with number"
                         + " of dimensions {}".format(num_size_dims))

    return torch.cat([probs.unsqueeze(0), logprobs.unsqueeze(0)], dim = 0)


def inputs2KL(inputs0 : Tuple[torch.Tensor], inputs1 : Tuple[torch.Tensor]) -> torch.Tensor:
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


def inputs2ent(inputs : Tuple[torch.Tensor]) -> torch.Tensor:
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


def inputs2acc(inputs: Tuple[torch.Tensor], samples: torch.Tensor) -> torch.Tensor:
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
    return (
        torch.where(samples == est_idx, 1.0, 1e-6).float()
        * torch.where(samples == -100, 0.0, 1.0).float()
    )


def logits2acc(logits : torch.Tensor, samples : torch.Tensor) -> torch.Tensor:
    """ Calculates and Returns the accuracy of a classification based
        on a batch of logits using a batch of samples. Please see
        the function inputs2acc to see how the accuracy is calculated
    """
    return inputs2acc(logits2inputs(logits), samples)


def logits2ent(logits : torch.Tensor) -> torch.Tensor:
    """ Calculates and Returns the entropies for a batch of multinomial
        distributions which are described by a set of logits. Please see
        the function inputs2ent to see how the entropy is calculated.
    """
    return inputs2ent(logits2inputs(logits))


def logits2KL(logits0 : torch.Tensor,logits1 : torch.Tensor) -> torch.Tensor:
    """ Calculates and Returns the KL divergence between two batches of
        multinomial distributions represented by logits. Please see
        the function inputs2KL to see how the KL divergence is calculated.
    """
    return inputs2KL(logits2inputs(logits0), logits2inputs(logits1))


def print_histogram(
    probs : torch.Tensor,
    labels : list,
    direction : bool=False,
    histogram_height : int=5
):
    """ Prints a block representation of a histogram to the user's terminal.

        Args:
            probs: the value of each bin in the histogram. This function
            assumes that all values should be normalized to a maximum
            to have a maximum magnitude of 1.0.

            labels: a list of strings, one for each bin, which act as
            labels in the print out.

            direction: a boolean which indicates whether a bin's value
            can be both negative or positive or just positive.

            histogram_height: the maximum number of rows used in the
            printed histogram, i.e. if the first bin has a value of 1.0
            and the histogram height is 10, then this function will
            print out a bar which takes up 10 rows in the terminal
    """
    block_length = 10 # magic number
    fill = "#"
    line = "-"
    gap = " "
    num_labels = len(labels)

    probs_clipped = torch.clamp(probs, -1, 1)

    counts = torch.round(probs_clipped * histogram_height).squeeze()

    if direction:
        lower_bound = -histogram_height-1
    else:
        lower_bound = 0

    for line_idx in range(histogram_height, lower_bound, -1):
        string = "   "

        for i in range(num_labels):
            count = counts[i]

            if count < line_idx and line_idx > 0:
                string += block_length * gap
                string += 3 * gap
            elif count >= line_idx and line_idx > 0:
                string += block_length * fill
                string += 3 * gap
            elif line_idx == 0:
                string += block_length * line
                string += 3 * line
            elif count >= line_idx and line_idx < 0:
                string += block_length * gap
                string += 3 * gap
            else:
                string += block_length * fill
                string += 3 * gap

        print(string)

    string = "   "

    for label in labels:
        remainder = block_length - len(label)

        if remainder % 2 == 0:
            offset = int(remainder / 2)
            string += ( offset * line + label + offset * line)
        else:
            offset = int((remainder - 1) / 2)
            string += ( (offset + 1) * line + label + offset * line)

        string += "   "

    print(string)

    string = "   "

    for i in range(num_labels):
        string += (block_length * line)
        string += "   "

    print(string)
    print("\n")
