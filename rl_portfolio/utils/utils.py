from __future__ import annotations

import copy
import torch

from random import randint
from random import random

from torch.utils.data.dataset import IterableDataset

from rl_portfolio.algorithm.buffers import GeometricReplayBuffer
from rl_portfolio.algorithm.buffers import SequentialReplayBuffer


class RLDataset(IterableDataset):
    def __init__(self, buffer, batch_size, sample_bias=1.0, from_start=False):
        """Initializes reinforcement learning dataset.

        Args:
            buffer: replay buffer to become iterable dataset.
            batch_size: Sample batch size. Not used if buffer is
                SequentialReplayBuffer.
            sample_bias: Probability of success of a trial in a geometric
                distribution. Only used if buffer is GeometricReplayBuffer.
            from_start: If True, will choose a sequence starting from the
                start of the buffer. Otherwise, it will start from the end.
                Only used if buffer is GeometricReplayBuffer.

        Note:
            It's a subclass of pytorch's IterableDataset,
            check https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        """
        self.buffer = buffer
        self.batch_size = batch_size
        self.sample_bias = 1.0
        self.from_start = from_start

    def __iter__(self):
        """Iterates over RLDataset.

        Returns:
          Every experience of a sample from replay buffer.
        """
        if isinstance(self.buffer, SequentialReplayBuffer):
            yield from self.buffer.sample()
        elif isinstance(self.buffer, GeometricReplayBuffer):
            yield from self.buffer.sample(
                self.batch_size, self.sample_bias, self.from_start
            )
        else:
            yield from self.buffer.sample(self.batch_size)


def apply_portfolio_noise(portfolio, epsilon=0.0):
    """Apply noise to portfolio distribution considering its constraints.

    Arg:
        portfolio: initial portfolio distribution.
        epsilon: maximum rebalancing.

    Returns:
        New portolio distribution with noise applied.
    """
    portfolio_size = portfolio.shape[0]
    new_portfolio = portfolio.copy()
    for i in range(portfolio_size):
        target_index = randint(0, portfolio_size - 1)
        difference = epsilon * random()
        # check constrains
        max_diff = min(new_portfolio[i], 1 - new_portfolio[target_index])
        difference = min(difference, max_diff)
        # apply difference
        new_portfolio[i] -= difference
        new_portfolio[target_index] += difference
    return new_portfolio


@torch.no_grad
def apply_parameter_noise(model, mean=0.0, std=0.0, device="cpu"):
    """Apply gaussian/normal noise to neural network.

    Arg:
        model: PyTorch model to add parameter noise.
        mean: Mean of gaussian/normal distribution.
        std: Standard deviation of gaussian/normal distribution.
        device: device of the model.

    Returns:
        Copy of model with parameter noise.
    """
    noise_model = copy.deepcopy(model)
    for param in noise_model.parameters():
        param += torch.normal(mean, std, size=param.shape).to(device)
    return noise_model
