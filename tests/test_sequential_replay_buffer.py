from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl_portfolio.algorithm.buffers import SequentialReplayBuffer

buffer = SequentialReplayBuffer(capacity=5)


def test_buffer_properties():
    """Tests replay buffer initial features."""
    assert len(buffer) == 0


def test_buffer_append():
    """Tests appending to replay buffer."""
    for i in range(20):
        buffer.append(i)

        assert len(buffer) == min(i + 1, 5)

        expected_buffer = []
        if i < 5:
            for j in range(i + 1):
                expected_buffer.append(j)
        else:
            for j in range(5):
                expected_buffer.append(i - (4 - j))

        assert list(buffer.buffer) == expected_buffer


def test_buffer_sample():
    """Tests if buffer samples sequential data and clears itself."""
    sample = buffer.sample()
    assert sample == [15, 16, 17, 18, 19]
    assert len(buffer) == 0
