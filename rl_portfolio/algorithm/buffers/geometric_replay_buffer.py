import numpy as np


class GeometricReplayBuffer:
    """This replay buffer saves the experiences of an RL agent in a deque
    (when buffer's capacity is full, it pops old experiences). When sampling
    from the buffer, a sequence of experiences will be chosen by sampling a
    geometric distribution that will favor more recent data.
    """

    def __init__(self, capacity):
        """Initializes geometric replay buffer.

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
        """Add experience to buffer. When buffer is full, it overwrites
        experiences in the beginning.

        Args:
            experience: experience to be saved.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (
                0 if self.position == self.capacity - 1 else self.position + 1
            )

    def add_at(self, experience, position):
        if isinstance(position, int):
            self.buffer[position] = experience
        if isinstance(position, list):
            assert isinstance(experience, list), "Experiences must also be a list."
            for exp, i in zip(experience, position):
                self.buffer[i] = exp

    def update_value(self, value, position, attr_or_index=None):
        if isinstance(position, int):
            if attr_or_index is None:
                self.buffer[position] = value
            else:
                self.buffer[position][attr_or_index] = value
        if isinstance(position, list):
            assert isinstance(value, list), "New values must also be a list."
            if attr_or_index is None:
                for val, pos in zip(value, position):
                    self.buffer[pos] = val
            else:
                for val, pos in zip(value, position):
                    item = list(self.buffer[pos])
                    item[attr_or_index] = val
                    self.buffer[pos] = tuple(item)

    def sample(self, batch_size, sample_bias=1.0, from_start=False):
        """REWRITE!!!!

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
            buffer = self.buffer[rand : rand + batch_size]
        else:
            buffer = self.buffer[max_pos - rand : max_pos - rand + batch_size]
        return buffer

    def reset(self):
        """Resets the replay buffer."""
        self.buffer = []
        self.position = 0
