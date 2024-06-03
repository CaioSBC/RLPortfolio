from collections import deque


class SequentialReplayBuffer:
    """This replay buffer saves the experiences of an RL agent in a deque
    (when buffer's capacity is full, it pops old experiences). When sampling
    from the buffer, all the experiences will be returned in order and the
    buffer will be cleared.
    """

    def __init__(self, capacity):
        """Initializes replay buffer.

        Args:
          capacity: Max capacity of buffer.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        """Represents the size of the buffer

        Returns:
          Size of the buffer.
        """
        return len(self.buffer)

    def append(self, experience):
        """Append experience to buffer. When buffer is full, it pops
           an old experience.

        Args:
          experience: experience to be saved.
        """
        self.buffer.append(experience)

    def sample(self):
        """Sample from replay buffer. All data from replay buffer is
        returned and the buffer is cleared.

        Returns:
          Sample of batch_size size.
        """
        buffer = list(self.buffer)
        self.buffer.clear()
        return buffer

    def reset(self):
        self.buffer = deque(maxlen=self.capacity)
