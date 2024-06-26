from __future__ import annotations

import copy
import torch

from torch.utils.data.dataset import IterableDataset

from rl_portfolio.algorithm.buffers import GeometricReplayBuffer


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
        self.sample_bias = sample_bias
        self.from_start = from_start

    def __iter__(self):
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


def apply_action_noise(actions, epsilon=0):
    """Apply noise to portfolio distribution considering its constraints.

    Arg:
        actions: Batch of agent actions.
        epsilon: Noise parameter.

    Returns:
        New batch of actions with applied noise.
    """
    # print("=======")
    if epsilon > 0:
        eps = 1e-7  # small value to avoid infinite numbers in log function
        log_actions = torch.log(actions + eps)
        # noise is calculated through a normal distribution with 0 mean and
        # std equal to the max absolute logarithmic value. Epsilon is used
        # to control the value of std.
        with torch.no_grad():
            noises = torch.normal(
                0,
                torch.max(torch.abs(log_actions), dim=1, keepdim=True)[0].expand_as(
                    log_actions
                )
                * epsilon,
            ).to(log_actions.device)
        new_actions = torch.softmax(log_actions + noises, dim=1)
        return new_actions
    else:
        return actions


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


def torch_to_numpy(tensor, squeeze=False):
    """Transforms torch tensor to numpy array.

    Arg:
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


def numpy_to_torch(array, type=torch.float32, add_batch_dim=False, device="cpu"):
    """Transforms numpy array to torch tensor.

    Arg:
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
