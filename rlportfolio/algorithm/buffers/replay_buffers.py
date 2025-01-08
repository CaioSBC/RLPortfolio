import random
import numpy as np

from collections import deque
from typing import Any


class ReplayBuffer:
    """Standard implementaton of replay buffer (when the buffer is full,
    old experiences are popped, like a deque). When sampling from the
    buffer, a random batch of experiences is returned."""

    def __init__(self, capacity):
        """Initializes replay buffer.

        Args:
            capacity: Max capacity of buffer.
        """
        self.capacity = capacity
        self.reset()

    def __len__(self):
        """Represents the size of the buffer.

        Returns:
            Size of the buffer.
        """
        return len(self.buffer)

    def add(self, experience):
        """Add experience to buffer. When the buffer is full, it pops
        an old experience.

        Args:
            experience: experience to be saved.
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Any]:
        """Sample from replay buffer. All data from replay buffer is
        returned and the buffer is cleared.

        Args:
            batch_size: Size of the sequential batch to be sampled.

        Returns:
            Sample of batch_size size.
        """
        sample = random.sample(self.buffer, batch_size)
        return sample

    def reset(self) -> None:
        """Resets the replay buffer."""
        self.buffer = deque(maxlen=self.capacity)

    def update_value(
        self, value: Any, position: int, attr_or_index: int | str | None = None
    ) -> None:
        """Updates the value of the item in a specific position of the
        replay buffer.

        Args:
            value: New value to be added to the buffer.
            position: Position of the item to be updated in the buffer.
            attr_or_index: If the item in the buffer are data structures
                like lists, tuples or dicts, this argument specifies which
                data to update.
        """
        if isinstance(position, int):
            if attr_or_index is None:
                self.buffer[position] = value
            else:
                if isinstance(self.buffer[position], tuple):
                    item = list(self.buffer[position])
                    item[attr_or_index] = value
                    self.buffer[position] = tuple(item)
                else:
                    self.buffer[position][attr_or_index] = value
        if isinstance(position, list):
            assert isinstance(value, list), "New values must also be a list."
            if attr_or_index is None:
                for val, pos in zip(value, position):
                    self.buffer[pos] = val
            else:
                for val, pos in zip(value, position):
                    if isinstance(self.buffer[pos], tuple):
                        item = list(self.buffer[pos])
                        item[attr_or_index] = val
                        self.buffer[pos] = tuple(item)
                    else:
                        self.buffer[pos][attr_or_index] = val


class ClearingReplayBuffer(ReplayBuffer):
    """This replay buffer acts like the standard one but, when sampling,
    it returns all its data and clears itself.
    """

    def sample(self) -> list[Any]:
        """Sample from replay buffer. All data from replay buffer is
        returned and the buffer is cleared.

        Returns:
            Sample of batch_size size.
        """
        sample = list(self.buffer)
        self.buffer.clear()
        return sample


class SequentialReplayBuffer(ReplayBuffer):
    """This replay buffer saves the experiences of an RL agent in a list
    (when buffer's capacity is full, it replaces values in the beginning
    of the list). When sampling from the buffer, a sequence of consecutive
    experiences will be randomly chosen.
    """

    def add(self, experience: Any) -> None:
        """Add experience to buffer. When buffer is full, it overwrites
        experiences in the beginning.

        Args:
            experience: Experience to be saved.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (
                0 if self.position == self.capacity - 1 else self.position + 1
            )

    def sample(self, batch_size: int) -> list[Any]:
        """Randomly samples a sequence of specified size from the replay buffer.

        Args:
            batch_size: Size of the sequential batch to be sampled.

        Returns:
          Sample of batch_size size.
        """
        max_pos = len(self.buffer) - batch_size
        # NOTE: we sum 1 to include the maximum position as a valid choice
        rand = np.random.randint(max_pos + 1)
        sample = self.buffer[rand : rand + batch_size]
        return sample

    def reset(self) -> None:
        """Resets the replay buffer."""
        self.buffer = []
        self.position = 0


class GeometricReplayBuffer(SequentialReplayBuffer):
    """This replay buffer saves the experiences of an RL agent in a list
    (when buffer's capacity is full, it replaces values in the beginning
    of the list). When sampling from the buffer, a sequence of consecutive
    experiences will be chosen by sampling a geometric distribution that
    will favor more recent data.
    """

    def sample(
        self, batch_size: int, sample_bias: float = 1.0, from_start: bool = False
    ) -> list[Any]:
        """Samples a sequence of specified size from the replay buffer. The
        sampling method will select the first item of the sequence following
        a geometric distribution, which, depending on the from_start argument,
        will favor samples from the beginning or from the end of the buffer.

        Args:
            batch_size: Size of the sequential batch to be sampled.
            sample_bias: Probability of success of a trial in a geometric
                distribution.
            from_start: If True, will choose a sequence starting from the
                start of the buffer. Otherwise, it will start from the end.

        Returns:
            Sample of batch_size size.
        """
        max_pos = len(self.buffer) - batch_size
        # NOTE: we subtract 1 so that rand can be 0 or the first/last
        # possible positions will be ignored.
        rand = np.random.geometric(sample_bias) - 1
        while rand > max_pos:
            rand = np.random.geometric(sample_bias) - 1
        if from_start:
            sample = self.buffer[rand : rand + batch_size]
        else:
            sample = self.buffer[max_pos - rand : max_pos - rand + batch_size]
        return sample
