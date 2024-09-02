from __future__ import annotations

from rlportfolio.algorithm.buffers import GeometricReplayBuffer

buffer = GeometricReplayBuffer(capacity=5)


def test_buffer_properties():
    """Tests replay buffer initial features."""
    assert len(buffer) == 0
    assert buffer.capacity == 5
    assert buffer.position == 0


def test_buffer_add():
    """Tests adding to replay buffer."""
    expected_buffer = []
    for i in range(20):
        buffer.add(i)

        assert len(buffer) == min(i + 1, 5)

        if i < 5:
            expected_buffer.append(i)
        else:
            index = i % 5
            expected_buffer[index] = i

        assert list(buffer.buffer) == expected_buffer


def test_buffer_sample():
    """Tests if buffer samples sequential data."""
    sample = buffer.sample(batch_size=3)
    assert sample == [17, 18, 19]
    assert len(buffer) == 5
    sample = buffer.sample(batch_size=3, from_start=True)
    assert sample == [15, 16, 17]
    assert len(buffer) == 5


def test_buffer_probabilistic_sample():
    """Tests if buffer samples valid sequences when using geometric
    distribution.
    """
    for i in range(20):
        sample = buffer.sample(batch_size=3, sample_bias=0.7)
        assert sample in [[15, 16, 17], [16, 17, 18], [17, 18, 19]]
