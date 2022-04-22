import enum
import torch
import torch.nn as nn
from types import FunctionType

import deep_learning.multinomial as multinomial
import deep_learning.gaussian as gaussian

class LossWrapper():
    """ A class which integrates logging the calculations of an loss
        function into the function
    """
    def __init__(self, loss_function : FunctionType):
        """ Inits an instance of LossWrapper """
        self.loss_function = loss_function

    def __call__(
        self,
        input_tuple : tuple,
        weight : float,
        logging_dict : dict,
        label : str
    ) -> torch.Tensor:
        """ Calculates the loss based on the 'input_tuple', stores
            the results in the 'logging_dict' using 'label' to generate
            its key and then returns the loss torch.Tensor
        """
        loss = weight * self.loss_function(*input_tuple)
        loss = loss[torch.nonzero(loss)]
        mean_loss = loss.mean()
        logging_dict['scalar'][label] = mean_loss.item()
        return mean_loss


class LossWrapperEnsemble(LossWrapper):
    """ A class which is basically the same as LossWrapper but
        targeted towards evaluating an ensemble of models.
    """
    def __call__(
        self,
        input_tuple : tuple,
        weight : float,
        logging_dict : dict,
        label : str
    ) -> torch.Tensor:
        """ Calculates the loss based on the 'input_tuple', stores
            the results in the 'logging_dict' using 'label' to generate
            its key and then returns the loss torch.Tensor for an
            ensemble of models
        """
        net_ests, target = input_tuple
        loss = torch.zeros(1).float().to(net_ests.device)

        for i in range(net_ests.size(0)):
            net_est = net_ests[i]
            ensemble_loss = weight * self.loss_function(net_est, target)
            ensemble_loss = ensemble_loss[torch.nonzero(ensemble_loss)]
            loss += ensemble_loss.mean()
        logging_dict['scalar'][label] = loss.item() / net_ests.size(0)
        return loss


class LossFactory(enum.Enum):
    L2 = LossWrapper(nn.MSELoss(reduction = "none"))
    L1 = LossWrapper(nn.L1Loss(reduction = "none"))
    L1Ensemble = LossWrapperEnsemble(nn.L1Loss(reduction = "none"))
    MultinomialNLL = LossWrapper(nn.CrossEntropyLoss(reduction = "none"))
    BinomialNLL = LossWrapper(nn.BCEWithLogitsLoss(reduction = "none"))
    MultinomialEntropy  = LossWrapper(multinomial.logits2ent)
    MultinomialKL = LossWrapper(multinomial.logits2KL)
    MultinomialKLEnsemble = LossWrapperEnsemble(multinomial.logits2KL)
    GaussianNLL = LossWrapper(gaussian.negative_log_likelihood)
    GaussianKL = LossWrapper(gaussian.divergence_KL)
    IdentityLoss = LossWrapper(lambda x : x)
