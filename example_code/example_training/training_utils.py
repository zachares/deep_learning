import enum
from deep_learning.losses import LossWrapper
from deep_learning.metrics import EvalMetricWrapper


class ProjectLossFactory(enum.Enum):
    example_loss = LossWrapper(lambda x : x)


class ProjectEvalMetricFactory(enum.Enum):
    example_metric = EvalMetricWrapper(lambda x : x)
