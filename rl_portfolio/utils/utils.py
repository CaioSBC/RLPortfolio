from __future__ import annotations

from random import randint
from random import random

from torch.utils.data.dataset import IterableDataset


class RLDataset(IterableDataset):
    def __init__(self, buffer):
        """Initializes reinforcement learning dataset.

        Args:
            buffer: replay buffer to become iterable dataset.

        Note:
            It's a subclass of pytorch's IterableDataset,
            check https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        """
        self.buffer = buffer

    def __iter__(self):
        """Iterates over RLDataset.

        Returns:
          Every experience of a sample from replay buffer.
        """
        yield from self.buffer.sample()


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