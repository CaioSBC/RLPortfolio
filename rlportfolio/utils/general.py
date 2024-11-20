from __future__ import annotations

import copy
import torch
import numpy as np

from collections.abc import Iterator
from typing import Any

from torch.utils.data.dataset import IterableDataset

from rlportfolio.algorithm.buffers import GeometricReplayBuffer, SequentialReplayBuffer
from rlportfolio.algorithm.buffers import PortfolioVectorMemory


class RLDataset(IterableDataset):
    def __init__(
        self,
        buffer: SequentialReplayBuffer,
        batch_size: int,
        sample_bias: float = 1.0,
        from_start: bool = False,
    ) -> RLDataset:
        """Initializes reinforcement learning dataset.

        Args:
            buffer: replay buffer to become iterable dataset.
            batch_size: Sample batch size.
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
        self.sample_bias = sample_bias
        self.from_start = from_start

    def __iter__(self) -> Iterator[Any]:
        """Iterates over RLDataset.

        Returns:
          Every experience of a sample from replay buffer.
        """
        if isinstance(self.buffer, GeometricReplayBuffer):
            yield from self.buffer.sample(
                self.batch_size, self.sample_bias, self.from_start
            )
        else:
            yield from self.buffer.sample(self.batch_size)


def apply_action_noise(
    actions: torch.Tensor,
    noise_model: str | None = None,
    epsilon: float = 0,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Apply noise to portfolio distribution considering its constraints.

    Args:
        actions: Batch of agent actions.
        noise_model: Name of the action noise model to use.
        epsilon: Random noise parameter.
        alpha: Alpha parameter for dirichlet distribution.

    Returns:
        New batch of actions with applied noise.
    """
    portfolio_size = actions.shape[1]

    if portfolio_size > 1:
        device = actions.device
        
        # To create action diversity, a random noise is applied to actions.
        if epsilon > 0 and noise_model is not None:
            if noise_model == "logarithmic":
                eps = 1e-7  # small value to avoid infinite numbers in log function
                log_actions = torch.log(actions + eps)
                # noise is calculated through a normal distribution with 0 mean and
                # std equal to the max absolute logarithmic value. Epsilon is used
                # to control the value of std.
                with torch.no_grad():
                    noises = torch.normal(
                        0,
                        torch.max(torch.abs(log_actions), dim=1, keepdim=True)[
                            0
                        ].expand_as(log_actions)
                        * epsilon,
                    ).to(device)
                noisy_actions = torch.softmax(log_actions + noises, dim=1)
            elif noise_model == "logarithmic_const":
                eps = 1e-7  # small value to avoid infinite numbers in log function
                log_actions = torch.log(actions + eps)
                # noise is calculated through a normal distribution with 0 mean and
                # std equal to ln(1e-7). Epsilon is used
                # to control the value of std.
                with torch.no_grad():
                    noises = torch.normal(
                        0,
                        np.abs(np.log(eps)) * torch.ones_like(log_actions) * epsilon,
                    ).to(device)
                noisy_actions = torch.softmax(log_actions + noises, dim=1)
            elif noise_model == "dirichlet":
                # noise is calculated based on the dirichlet distribution. A random
                # action tensor is created and epsilon is used to define the weight
                # of the randomness.
                with torch.no_grad():
                    random_actions = torch.distributions.Dirichlet(
                        torch.ones_like(actions) * alpha
                    ).sample()
                    noises = epsilon * (random_actions - actions)
                noisy_actions = actions + noises
            else:
                raise ValueError("Inexistent noise model: {}".format(noise_model))
        else:
            noisy_actions = actions
        return noisy_actions
    else:
        return actions


@torch.no_grad
def apply_parameter_noise(
    model: torch.nn.Module, epsilon: float = 0
) -> torch.nn.Module:
    """Apply noise to PyTorch model. If the model is a portfolio optimization
    policy, the noise allows the agent to generate different actions and
    explore the action space.

    Args:
        model: PyTorch model to add parameter noise.
        epsilon: Noise parameter.

    Returns:
        Copy of model with parameter noise.
    """
    if epsilon > 0:
        noisy_model = copy.deepcopy(model)
        for param in noisy_model.parameters():
            param = param + torch.normal(
                0, torch.abs(param) * epsilon, size=param.shape
            ).to(param.device)
        return noisy_model
    else:
        return model


def torch_to_numpy(tensor: torch.Tensor, squeeze: float = False) -> np.ndarray:
    """Transforms torch tensor to numpy array.

    Args:
        tensor: Tensor to be transformed.
        squeeze: If True, numpy array will be squeezed, eliminating
            dimensions of size 1.

    Returns:
        Numpy array.
    """
    array = tensor.cpu().detach().numpy()
    if squeeze:
        array = array.squeeze()
    return array


def numpy_to_torch(
    array: np.ndarray,
    type: torch.dtype = torch.float32,
    add_batch_dim: bool = False,
    device: str = "cpu",
) -> torch.Tensor:
    """Transforms numpy array to torch tensor.

    Args:
        array: Numpy array to be transformed.
        type: Type of torch tensor.
        device: Torch tensor device.

    Returns:
        Torch tensor.
    """
    tensor = torch.from_numpy(array).to(type).to(device)
    if add_batch_dim:
        tensor = tensor.unsqueeze(dim=0)
    return tensor


def combine_replay_buffers(
    rb_list: list[SequentialReplayBuffer], new_type: type[SequentialReplayBuffer]
) -> SequentialReplayBuffer:
    """Combines multiple replay buffers and creates a new one.

    Args:
        rb_list: List of replay buffers.
        new_type: New replay buffer type. It can be SequentialReplayBuffer or
            GeometricReplayBuffer.

    Note:
        After combining replay buffers, its position pointer will be reset to 0
        so it is adviseable to avoid combining replay buffers if the integrity
        of the position pointer is important to the algorithm.

    Returns:
        Combined replay buffer.
    """
    new_capacity = 0
    new_buffer = []
    for rb in rb_list:
        new_capacity += rb.capacity
        new_buffer += rb.buffer
    new_rb = new_type(new_capacity)
    new_rb.buffer = new_buffer
    return new_rb


def combine_portfolio_vector_memories(
    pvm_list: list[PortfolioVectorMemory], move_index: bool = True
) -> PortfolioVectorMemory:
    """Combines two portfolio vector memories and creates a new one.

    Args:
        pvm_list: List of portfolio vector memories.
        move_index: If True, moves the index pointer of the portfolio vector
            memory to the end, so it appends new experiences.

    Returns:
        Combined portfolio vector memory.
    """
    new_capacity = 0
    new_memory = []
    new_index = 0
    new_portfolio_size = 0
    for i in range(len(pvm_list)):
        if i > 0:
            assert (
                pvm_list[i].portfolio_size == new_portfolio_size
            ), "Portfolio vector memories must have the same portfolio size."
        else:
            new_portfolio_size = pvm_list[i].portfolio_size
        new_index = max(new_capacity, 0) if move_index else 0
        new_capacity += pvm_list[i].capacity
        new_memory += pvm_list[i].memory if i == 0 else pvm_list[i].memory[1:]
    new_pvm = PortfolioVectorMemory(new_capacity, new_portfolio_size)
    new_pvm.memory = new_memory
    new_pvm.index = new_index
    return new_pvm
