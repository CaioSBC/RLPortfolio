import numpy as np

from collections import deque
from itertools import islice


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
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        """Represents the size of the buffer.

        Returns:
          Size of the buffer.
        """
        return len(self.buffer)

    def append(self, experience):
        """Append experience to buffer. When buffer is full, it pops an old
        experience.

        Args:
          experience: experience to be saved.
        """
        self.buffer.append(experience)

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
            buffer = list(islice(self.buffer, rand, rand + batch_size))
        else:
            buffer = list(
                islice(self.buffer, max_index - rand, max_index - rand + batch_size)
            )
        return buffer

    def reset(self):
        self.buffer = deque(maxlen=self.capacity)
