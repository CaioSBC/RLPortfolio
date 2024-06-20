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

    def add(self, experience):
        """Add experience to buffer. When buffer is full, it pops
           an old experience.

        Args:
          experience: experience to be saved.
        """
        self.buffer.append(experience)

    def add_at(self, experience, index):
        if isinstance(index, int):
            self.buffer[index] = experience
        if isinstance(index, list):
            assert isinstance(experience, list), "Experiences must also be a list."
            for exp, i in zip(experience, index):
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
