from __future__ import annotations

import numpy as np

from typing import Any


class SequentialReplayBuffer:
    """This replay buffer saves the experiences of an RL agent in a list
    (when buffer's capacity is full, it replaces values in the beginning
    of the list). When sampling from the buffer, a sequence of consecutive
    experiences will be randomly chosen.
    """

    def __init__(self, capacity: int) -> SequentialReplayBuffer:
        """Initializes the replay buffer.

        Args:
            capacity: Max capacity of buffer.
        """
        self.capacity = capacity
        self.reset()

    def __len__(self) -> int:
        """Represents the size of the buffer.

        Returns:
            Size of the buffer.
        """
        return len(self.buffer)

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

    def sample(self, batch_size: int) -> list[Any]:
        """Randomly samples a sequence of specified size from the replay buffer.

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
