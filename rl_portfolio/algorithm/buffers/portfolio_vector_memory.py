import numpy as np

class PortfolioVectorMemory:
    def __init__(self, capacity, portfolio_size):
        """Initializes portfolio vector memory.

        Args:
          capacity: Max capacity of memory.
          portfolio_size: Portfolio size.
        """
        # initially, memory will have the same actions
        self.capacity = capacity
        self.portfolio_size = portfolio_size
        self.reset()

    def reset(self):
        self.memory = [np.array([1] + [0] * self.portfolio_size, dtype=np.float32)] * (
            self.capacity + 1
        )
        self.index = 0  # initial index to retrieve data

    def retrieve(self):
        last_action = self.memory[self.index]
        self.index = 0 if self.index == self.capacity else self.index + 1
        return last_action

    def add(self, action):
        self.memory[self.index] = action