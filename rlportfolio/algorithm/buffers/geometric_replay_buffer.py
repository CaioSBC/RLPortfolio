from __future__ import annotations

import numpy as np

from rlportfolio.algorithm.buffers import SequentialReplayBuffer

from typing import Any


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
