from __future__ import annotations

from rlportfolio.algorithm.buffers import SequentialReplayBuffer

buffer = SequentialReplayBuffer(capacity=5)


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
    for i in range(20):
        sample = buffer.sample(batch_size=3)
        assert sample in [[15, 16, 17], [16, 17, 18], [17, 18, 19]]
        assert len(sample) == 3


def test_buffer_update_value():
    """Tests if the buffer is able to update a value at a specific
    position.
    """
    buffer.update_value(0, 3)
    assert buffer.buffer == [15, 16, 17, 0, 19]
    buffer.update_value(7, 0)
    assert buffer.buffer == [7, 16, 17, 0, 19]
    buffer.update_value(3, 4)
    assert buffer.buffer == [7, 16, 17, 0, 3]


def test_buffer_update_multiple_values():
    """Tests if the buffer is able to update several values at specific
    positions.
    """
    buffer.update_value([3, 9, 12], [3, 0, 4])
    assert buffer.buffer == [9, 16, 17, 3, 12]


def test_buffer_update_value_tuple():
    """Tests if the buffer is able to update a value at a specific
    position if the experience is a tuple.
    """
    buffer.reset()
    for i in range(5):
        buffer.add((i, 2 * i))
    assert buffer.buffer == [(0, 0), (1, 2), (2, 4), (3, 6), (4, 8)]
    buffer.update_value(3, 0, 0)
    assert buffer.buffer == [(3, 0), (1, 2), (2, 4), (3, 6), (4, 8)]
    buffer.update_value(2, 2, 1)
    assert buffer.buffer == [(3, 0), (1, 2), (2, 2), (3, 6), (4, 8)]
    buffer.update_value(5, 4, 1)
    assert buffer.buffer == [(3, 0), (1, 2), (2, 2), (3, 6), (4, 5)]


def test_buffer_update_multiple_values_tuple():
    """Tests if the buffer is able to update several value at specific
    positions if the experience is a tuple.
    """
    buffer.reset()
    for i in range(5):
        buffer.add((i, 2 * i))
    assert buffer.buffer == [(0, 0), (1, 2), (2, 4), (3, 6), (4, 8)]
    buffer.update_value([3, 2, 1], [0, 2, 4], 0)
    assert buffer.buffer == [(3, 0), (1, 2), (2, 4), (3, 6), (1, 8)]


def test_buffer_update_value_dict():
    """Tests if the buffer is able to update a value at a specific
    position if the experience is a dict.
    """
    buffer.reset()
    for i in range(5):
        buffer.add({"key_1": i, "key_2": 2 * i})
    assert buffer.buffer == [
        {"key_1": 0, "key_2": 0},
        {"key_1": 1, "key_2": 2},
        {"key_1": 2, "key_2": 4},
        {"key_1": 3, "key_2": 6},
        {"key_1": 4, "key_2": 8},
    ]
    buffer.update_value(3, 0, "key_1")
    assert buffer.buffer == [
        {"key_1": 3, "key_2": 0},
        {"key_1": 1, "key_2": 2},
        {"key_1": 2, "key_2": 4},
        {"key_1": 3, "key_2": 6},
        {"key_1": 4, "key_2": 8},
    ]
    buffer.update_value(2, 2, "key_2")
    assert buffer.buffer == [
        {"key_1": 3, "key_2": 0},
        {"key_1": 1, "key_2": 2},
        {"key_1": 2, "key_2": 2},
        {"key_1": 3, "key_2": 6},
        {"key_1": 4, "key_2": 8},
    ]
    buffer.update_value(5, 4, "key_2")
    assert buffer.buffer == [
        {"key_1": 3, "key_2": 0},
        {"key_1": 1, "key_2": 2},
        {"key_1": 2, "key_2": 2},
        {"key_1": 3, "key_2": 6},
        {"key_1": 4, "key_2": 5},
    ]


def test_buffer_update_multiple_values_dict():
    """Tests if the buffer is able to update several values at specific
    positions if the experience is a dict.
    """
    buffer.reset()
    for i in range(5):
        buffer.add({"key_1": i, "key_2": 2 * i})
    assert buffer.buffer == [
        {"key_1": 0, "key_2": 0},
        {"key_1": 1, "key_2": 2},
        {"key_1": 2, "key_2": 4},
        {"key_1": 3, "key_2": 6},
        {"key_1": 4, "key_2": 8},
    ]
    buffer.update_value([3, 2, 5], [0, 2, 4], "key_2")
    assert buffer.buffer == [
        {"key_1": 0, "key_2": 3},
        {"key_1": 1, "key_2": 2},
        {"key_1": 2, "key_2": 2},
        {"key_1": 3, "key_2": 6},
        {"key_1": 4, "key_2": 5},
    ]
