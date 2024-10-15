import logging
from logging import getLogger

import numpy as np
import torch.optim as optim
from recbole.data.dataloader.general_dataloader import AbstractDataLoader

logger = logging.getLogger(__name__)


class FeatureDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, features, batch_size, sampler, shuffle=False):
        self.logger = getLogger()
        self.sample_size = len(dataset)
        self.features = features
        self.batch_size = batch_size
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.batch_size
        self.step = batch_size
        self.set_batch_size(batch_size)

    def collate_fn(self, index):
        index = np.array(index)
        return self._dataset[index], self.features[index]


def build_optimizer(**kwargs):
    r"""Init the Optimizer
    Args:
        params (torch.nn.Parameter, optional): The parameters to be optimized.
        learner (str, optional): The name of used optimizer. Defaults to ``adamw``.
        learning_rate (float, optional): Learning rate. Defaults to ``1e-2``.
        weight_decay (float, optional): The L2 regularization weight. Defaults to ``1e-6``.
    Returns:
        torch.optim: the optimizer
    """
    params = None
    learner = None
    weight_decay = None
    learning_rate = None
    params = kwargs.pop("params", params)
    learner = kwargs.pop("learner", learner)
    learning_rate = kwargs.pop("learning_rate", learning_rate)
    weight_decay = kwargs.pop("weight_decay", weight_decay)

    if learner.lower() == "adam":
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif learner.lower() == "adamw":
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif learner.lower() == "sgd":
        optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
    elif learner.lower() == "adagrad":
        optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
    elif learner.lower() == "rmsprop":
        optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
    elif learner.lower() == "sparse_adam":
        optimizer = optim.SparseAdam(params, lr=learning_rate)
        if weight_decay > 0:
            logger.warning("Sparse Adam cannot argument received argument [{weight_decay}]")
    else:
        logger.warning("Received unrecognized optimizer, set default Adam optimizer")
        optimizer = optim.Adam(params, lr=learning_rate)

    return optimizer


def average(inputs: list[int | float | list | dict], std: bool = False):
    if isinstance(inputs[0], (int, float)):
        if std:
            return (np.mean(inputs), np.std(inputs))
        else:
            return np.mean(inputs)
    elif isinstance(inputs[0], list):
        return [average([*ls], std=std) for ls in zip(*inputs)]
    elif isinstance(inputs[0], dict):
        return {k: average([dc[k] for dc in inputs], std=std) for k in inputs[0].keys()}
    else:
        raise TypeError
