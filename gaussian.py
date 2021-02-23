import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple

def logprobs(means: torch.Tensor,
             precs: torch.Tensor, 
             samples: torch.Tensor) -> torch.Tensor:
    """ Calculates the log likelihoods of a batch of sample tensors
        according to the parameters of Gaussian distributions described 
        by mean and precision tensors

        Note: the precision matrix of a Gaussian distribution is the 
        inverse of its covariance matrix. The precision matrix is used
        instead of the covariance matrix because then it is possible
        to calculate the log likelihood of a sample without performing
        a matrix inversion

        Args:
            means: a torch.Tensor describing the mean parameters of a
            batch of Gaussian distributions; can be of size N, NxM
            precs: a torch.Tensor describing the precision matrices of a
            batch of Gaussian distributions; can be of size N, NxM,
            NxMxM
            samples: a torch.Tensor of sample values; can be of size N,
            NxM

            Note: N = batch size, M = vector size. A precs tensor of
            size N indicates the input distributions are univariate
            Gaussian distributions. A precs tensor of size NxM indicates
            the input distribution is either multiple univariate 
            Gaussian distributions or a multivariate Gaussian
            distribution with a diagonal covariance matrix. This code
            assumes the former. A precs tensor of size NxMxM indicates 
            the input distribution is a multivariate Gaussian
            distribution.
        
        Returns:
            A torch.Tensor of loglikelihoods for the inputted samples
            based on the inputted Gaussian parameters
        
        Raises:
            ValueError: if the rank of the argument prec torch.Tensor 
            does not match any of the ranks the function is compatible with
    """
    # Calculates the rank of the precision tensor
    num_size_dims = len(list(precs.size()))

    if num_size_dims == 1:
        return 0.5 * (torch.log(precs) - precs * (samples - means).pow(2)
        - (np.log(2 * np.pi)))
    elif num_size_dims == 2:
        # multiple univariate Gaussian distributions
        return 0.5 * (torch.log(precs) - precs * (samples - means).pow(2)
        - np.log(2 * np.pi)).sum(-1)

        # multivariate Gaussian distributions with diagonal covariance
        # matrices
        # return 0.5 * (torch.log(precs.prod(1)) - precs * (samples
        # - means).pow(2) - precs.size(1) * np.log(2 * np.pi)).sum(-1)
    elif num_size_dims == 3:
        log_prob_const = 0.5 * (torch.log(torch.det(precs))
        - precs.size(1) * np.log(2 * np.pi))

        err = (samples - means).unsqueeze(1) 

        log_prob_sample = -0.5 * torch.bmm(torch.bmm(err, precs),\
            err.transpose(1,2))

        return (log_prob_const + log_prob_sample).squeeze()
    else:
        raise ValueError("logprob function does not support tensors of\
             rank {}".format(num_size_dims))

def negative_log_likelihood(params: Tuple[torch.Tensor],
                            samples: torch.Tensor) -> torch.Tensor:
    """ Calculates the negative loglikelihoods for a batch of samples
        using a batch of parameters for Gaussian distributions
        
        Args:
            params: a tuple with two torch.Tensor arrays, the first
            array in the tuple is the mean parameters and the second
            array is the precision matrices of the Gaussian distributions
            used to evaluate the samples
            samples: a torch.Tensor array with sample values to evaluate
        
        Returns:
            A torch.Tensor array of size N, where N is the batch size of
            the samples argument
    """
    means, precs = params
    return -1.0 * logprobs(means, precs, samples)

def divergence_KL(params0: Tuple[torch.Tensor],
                  params1: Tuple[torch.Tensor]) -> torch.Tensor:
    """ Calculates the Kullback-Leibler Divergence between two batches
        of Gaussian distributions

        Args:
            params0: a tuple with two torch.Tensor arrays, the first
            array in the tuple is the mean parameters and the second
            array is the precision matrices
            params1: a tuple with two torch.Tensor arrays, the first
            array in the tuple is the mean parameters and the second
            array is the precision matrices

            Note: the precs0 and precs1 torch.Tensors can be of size N, 
            NxM or NxMxM. N = batch size, M = vectorsize. A precs tensor
            of size N indicates the input distributions are univariate
            Gaussian distributions. A precs tensor of size NxM indicates
            the input distribution is either multiple univariate 
            Gaussian distributions or a multivariate Gaussian
            distribution with a diagonal covariance matrix. This code
            assumes the former. A precs tensor of size NxMxM indicates 
            the input distribution is a multivariate Gaussian
            distribution.

        Returns:
            A torch.Tensor of size N of KL Divergence values 
            for the two distributions
        
        Raises:
            ValueError: if the rank of the inputted prec torch.Tensor 
            does not match any of the ranks the function is compatible with
    """
    means0, precs0 = params0
    means1, precs1 = params1

    # Calculates the rank of the precision tensors
    num_size_dims = len(list(precs0.size()))

    if num_size_dims == 1:
        return 0.5 * (torch.log(precs0) - torch.log(precs1) + precs1 / precs0
        + precs1 * (means0 - means1).pow(2) - 1)
    elif num_size_dims == 2:
        # multiple univariate Gaussian distributions
        return 0.5 * (torch.log(precs0) - torch.log(precs1) + precs1 / precs0
        + precs1 * (means0 - means1).pow(2) - 1).sum(-1)

        # multivariate Gaussian distributions with diagonal covariance
        # matrices
        # return 0.5 * (torch.log(precs0.prod(1)) - torch.log(precs1.prod(1))
        # + precs1 / precs0 + precs1 * (means0 - means1).pow(2) - 1).sum(-1)
    elif num_size_dims == 3:
        precs_dets0 = torch.det(precs0)
        precs_dets1 = torch.det(precs1)

        precs0_inv = torch.inverse(precs0)
        precs_mult = torch.bmm(precs1, precs0_inv)

        precs_mult_trace = torch.diagonal(precs_mult, dim1=1, dim2=2).sum(1)

        mean_error = (means1 - means0).unsqueeze(2)

        return 0.5 * (torch.log(precs_dets0/precs_dets1) + precs_mult_trace
        + torch.bmm(torch.bmm(mean_error.transpose(1,2), precs1),
        mean_error).squeeze())
    else:
        raise ValueError("estimates tensor size invalid with number of\
             dimensions {}".format(num_size_dims))

# TODO add sampling for a Multivariate Normal Distribution
def sample(means: torch.Tensor,
           precs: torch.Tensor, 
           device: torch.device) -> torch.Tensor:
    """ Samples a batch of Gaussian distributions

        Note: this sampling function takes advantage of the reparametrization 
        trick used in variational autoencoders to make sampling a univariate
        Gaussian distribution differentiable. There is a reparametrization
        trick for multivariate Gaussian distributions, but has not be 
        implemented yet in this function

        Args:
            - means: torch.Tensor of the means for each Gaussian
            distribution in the batch
            - precs: torch.Tensor of the precision matrices of each
            Gaussian distribution in the batch
        
        Returns:
            - torch.Tensor of sample values, the tensor is the same size
            as the argument means
        
        Raises:
            - ValueError: if the precision tensor is of rank 3, which is
            currently not supported but will be implemented in future
            version of the code
            - ValueError: if the precision tensor is of a rank other
            than 1,2 or 3 in which case this function cannot sample the
            input batch of distributions
    """
    # Calculates the rank of the precision matrices tensor
    num_size_dims = len(list(precs.size()))
    
    # the reparametrization trick for a univariate Gaussian distribution
    if num_size_dims == 1 or num_size_dims == 2:
        variance = 1 / (F.softplus(precs) + 1e-8)
        epsilon = Normal(0, 1).sample(means.size())
        return means + torch.sqrt(variance) * epsilon.to(device)
    elif num_size_dims == 3:
        raise ValueError("multivariate Gaussian\
            Distributions are not currently supported")        
    else:
        raise ValueError("invalid size for precision matrices of {}"\
            .format(num_size_dims))

def output2params(params: torch.Tensor) -> Tuple[torch.Tensor]:
    """ Converts a tensor of size Nx2M to two tensors of size NxM, where
        N is the batch size of the tensor and M is the number of
        components in the mean vector

        Note: this function comes in handy if you want to convert the 
        output of a network to a set of means and variances.

        Args:
            params: the torch.Tensor to be split and processed into a
            set of Gaussian parameters
        
        Returns:
            - a tuple with two elements; the first element is a 
            torch.Tensor of mean parameters for Gaussian distributions
            and the second element is a torch.Tensor of precision
            matrices for Gaussian distributions
        
        Raises:
            - ValueError: when the argument params has a rank which is
            not supported by this function
    """
    num_size_dims = len(list(params.size()))

    if num_size_dims == 2:
        means, precs_unprocessed = torch.split(params, params.size(1) // 2, dim=1) 
        return means, precs_unprocessed.pow(2)
    else:
        raise ValueError("params tensor has a rank of {}, which is not\
             supported by this function".format(num_size_dims))

def params2prec(params : torch.Tensor) -> torch.Tensor:
    """ Calculates a precision matrix for a multivariate Gaussian 
        distribution from a real valued vector. 
        
        This function relies on the cholesky decomposition of the 
        precision matrix which is a lower triangular matrix with 
        positive diagonal values.

        Args:
            params : a torch.Tensor of size Nx(1 + 2 + .... + M) where
            M is the dimension of the multivariate Gaussian distribution

        Returns:
            a torch.Tensor of size NxMxM describing the precision
            matrices for a batch of N inputs.
        
        Raises:
            ValueError: if the input torch.Tensor is not the correct
            rank
    """

    if len(list(params.size())) == 2:
        m_tril = params.tril()

        m_tril[torch.arange(m_tril.size(0)), torch.arange(m_tril.size(1))] =\
             torch.abs(torch.diagonal(m_tril, dim1 = 1, dim2=2))

        prec = torch.bmm(m_tril.transpose(1,2).transpose(1,0),
                         m_tril.transpose(1,2).transpose(1,0).transpose(1,2))   
    else:
        raise ValueError("The rank of the tensor {} is not 2"
                         .format(len(list(params.size()))))

    return prec

def det3(mats: torch.Tensor) -> torch.Tensor:
    """ Calculates the determinant for a batch of 3x3 matrices

        Args:
            mats: a torch.Tensor of size Nx3x3 which is a batch of 3x3 
            matrices
        
        Returns:
            a torch.Tensor of size N with the determinant for each
            matrix in the batch
        
        Raises:
            ValueError: if the input tensor is of the incorrect rank     
    """
    
    num_size_dims = len(list(mats.size()))

    if num_size_dims != 3:
        raise ValueError("mats tensor has a rank of {}, which is not\
             supported by this function".format(num_size_dims))
    else:
        return mats[:,0,0] * (mats[:,1,1] * mats[:,2,2] - mats[:,1,2] *mats[:,2,1]) -\
        mats[:,0,1] * (mats[:,1,0] * mats[:,2,2] - mats[:,2,0] * mats[:,1,2]) +\
        mats[:,0,2] * (mats[:,1,0] * mats[:,2,1] - mats[:,1,1] * mats[:,2,0])

#### TODO debug product of Gaussians function
# def productofgaussians(mean_vect, prec_vect):
#     num_size_dims = len(list(mean_vect.size()))

#     if num_size_dims == 2:
#         prec = prec_vect.sum(1)
#         mean = mean_vect * prec_vect / prec.unsqueeze(1).repeat_interleave(prec_vect.size(1), dim = 1)
        
#         return mean, prec
#     elif num_size_dims == 3:
#         prec = prec_vect.sum(1)

#         mean = (mean_vect * prec_vect / prec.unsqueeze(1).repeat_interleave(prec_vect.size(1), dim = 1)).sum(1)

#         return mean, prec

#     elif num_size_dims == 4:
#         prec = prec_vect.sum(1)

#         prec_inv_reshape = torch.reshape(torch.inverse(prec).unsqueeze(1).repeat_interleave(prec_vect.size(1), dim = 1), (prec.size(0) * prec_vect.size(1), prec.size(1), prec.size(2)))
#         prec_vect_reshape = torch.reshape(prec_vect, (prec_vect.size(0) * prec_vect.size(1), prec_vect.size(2), prec_vect.size(3)))
#         prec_mm_reshape = torch.bmm(prec_inv_reshape, prec_vect_reshape)
#         mean_vect_reshape = torch.reshape(mean_vect, (mean_vect.size(0) * mean_vect.size(1), mean_vect.size(2), 1))
#         mean_unsummed_reshape = torch.bmm(prec_mm_reshape, mean_vect_reshape)
#         mean = torch.reshape(mean_unsummed_reshape, (mean_vect.size(0), mean_vect.size(1), mean_vect.size(2))).sum(1)

#         return mean, prec   
#     else:
#         raise Exception("estimates tensor size invalid with number of dimensions" + str(num_size_dims))