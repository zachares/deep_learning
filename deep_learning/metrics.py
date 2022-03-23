import enum
import torch
from types import FunctionType

import deep_learning.multinomial as multinomial

class EvalMetricWrapper():
    """ A class which integrates logging the calculations of an
        evaluation function into the function
    """
    def __init__(self, metric_function : FunctionType):
        """ Inits an instance of EvalMetricWrapper """
        self.metric = metric_function

    def __call__(self, input_tuple : tuple, logging_dict : dict, label : str):
        """ Calculates an evaluation metric based on the 'input_tuple' and
            stores the results in the 'logging_dict' using 'label' to
            generate its key
        """
        measurement = self.metric(*input_tuple)
        logging_dict['scalar'][label] = measurement.mean().item()


class EvalMetricWrapperEnsemble(EvalMetricWrapper):
    """ A class which is basically the same as EvalMetricWrapper but
        targeted towards evaluating an ensemble of models.
    """
    def __call__(self, input_tuple : tuple, logging_dict : dict, label : str):
        net_ests, target = input_tuple
        for i in range(net_ests.size(0)):
            net_est = net_ests[i]
            measurement = self.metric(net_est, target)
            key = f"{label}_{i}"
            logging_dict['scalar'][key] = measurement.mean().item()


def continuous2error_mag(ests : torch.Tensor, gt_values : torch.Tensor) -> torch.Tensor:
    """ Calculates the magnitude of the error between two real vectors
        in an estimation problem."""
    num_size_dims = len(list(ests.size()))
    errors = gt_values - ests
    assert num_size_dims >= 4
    assert ests.shape[-1] == gt_values.shape[-1]
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


def binomial_accuracy(logits : torch.Tensor, labels : torch.Tensor) -> torch.Tensor:
    """ Calculates the accuracy of a binomial distributions parameters
        based on a single sample
    """
    probs = torch.sigmoid(logits)
    samples = torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs))
    accuracy = torch.where(samples == labels, torch.ones_like(samples), torch.zeros_like(samples))
    return accuracy


class EvalMetricFactory(enum.Enum):
    MultinomialAccuracy = EvalMetricWrapper(multinomial.logits2acc)
    MultinomialAccuracyEnsemble = EvalMetricWrapperEnsemble(multinomial.logits2acc)
    MultinomialEntropy = EvalMetricWrapper(multinomial.logits2ent)
    MultinomialEntropyEnsemble = EvalMetricWrapperEnsemble(multinomial.logits2ent)
    ContinuousErrorMag = EvalMetricWrapper(continuous2error_mag)
    ContinuousErrorMagEnsemble = EvalMetricWrapperEnsemble(continuous2error_mag)
    BinomialAccuracy = EvalMetricWrapper(binomial_accuracy)
