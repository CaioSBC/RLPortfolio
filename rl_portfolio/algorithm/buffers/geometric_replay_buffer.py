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

    def append(self, experience):
        """Append experience to buffer. When buffer is full, it overwrites
        experiences in the beginning.

        Args:
            experience: experience to be saved.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.index] = experience
            self.index = 0 if self.index == self.capacity - 1 else self.index + 1

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
        max_index = len(self.buffer) - batch_size
        # NOTE: we subtract 1 so that rand can be 0 or the first/last
        # possible indexes will be ignored.
        rand = np.random.geometric(sample_bias) - 1
        while rand > max_index:
            rand = np.random.geometric(sample_bias) - 1
        if from_start:
            buffer = self.buffer[rand : rand + batch_size]
        else:
            buffer = self.buffer[max_index - rand : max_index - rand + batch_size]
        return buffer

    def reset(self):
        """Resets the replay buffer."""
        self.buffer = []
        self.index = 0
