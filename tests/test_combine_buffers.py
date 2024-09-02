from __future__ import annotations

import numpy as np
import pytest

from rlportfolio.algorithm.buffers import PortfolioVectorMemory
from rlportfolio.algorithm.buffers import SequentialReplayBuffer
from rlportfolio.utils import combine_portfolio_vector_memories
from rlportfolio.utils import combine_replay_buffers


def test_combine_two_rbs_1():
    """Tests if two full replay buffers are effectively combined."""
    rb_1 = SequentialReplayBuffer(capacity=5)
    rb_2 = SequentialReplayBuffer(capacity=3)
    for i in range(5):
        rb_1.add(i)
        if i < 3:
            rb_2.add(i)
    assert rb_1.buffer == [0, 1, 2, 3, 4]
    assert rb_2.buffer == [0, 1, 2]
    new_rb = combine_replay_buffers([rb_1, rb_2], SequentialReplayBuffer)
    assert new_rb.capacity == 8
    assert new_rb.position == 0
    assert new_rb.buffer == [0, 1, 2, 3, 4, 0, 1, 2]
    new_rb.add(3)
    new_rb.add(4)
    new_rb.add(5)
    assert new_rb.position == 3
    assert new_rb.buffer == [3, 4, 5, 3, 4, 0, 1, 2]


def test_combine_two_rbs_2():
    """Tests if two replay buffers (on full and other not full) are
    effectively combined."""
    rb_1 = SequentialReplayBuffer(capacity=5)
    rb_2 = SequentialReplayBuffer(capacity=5)
    for i in range(5):
        rb_1.add(i)
        if i < 3:
            rb_2.add(i)
    assert rb_1.buffer == [0, 1, 2, 3, 4]
    assert rb_2.buffer == [0, 1, 2]
    new_rb = combine_replay_buffers([rb_1, rb_2], SequentialReplayBuffer)
    assert new_rb.capacity == 10
    assert new_rb.position == 0
    assert new_rb.buffer == [0, 1, 2, 3, 4, 0, 1, 2]
    new_rb.add(3)
    new_rb.add(4)
    new_rb.add(5)
    assert new_rb.position == 1
    assert new_rb.buffer == [5, 1, 2, 3, 4, 0, 1, 2, 3, 4]


def test_combine_multiple_rbs():
    """Tests if several replay buffers will be combined correctly."""
    rb_list = [
        SequentialReplayBuffer(capacity=5),
        SequentialReplayBuffer(capacity=3),
        SequentialReplayBuffer(capacity=7),
        SequentialReplayBuffer(capacity=2),
    ]
    for rb in rb_list:
        for i in range(5):
            rb.add(i)
    new_rb = combine_replay_buffers(rb_list, SequentialReplayBuffer)
    assert new_rb.capacity == 17
    assert new_rb.position == 0
    assert new_rb.buffer == [0, 1, 2, 3, 4, 3, 4, 2, 0, 1, 2, 3, 4, 4, 3]
    new_rb.add(3)
    new_rb.add(4)
    new_rb.add(5)
    assert new_rb.position == 1
    assert new_rb.buffer == [5, 1, 2, 3, 4, 3, 4, 2, 0, 1, 2, 3, 4, 4, 3, 3, 4]


def test_combine_two_pvms():
    """Tests if two portfolio vector memories are being correctly combined."""
    pvm_1 = PortfolioVectorMemory(3, 3)
    pvm_2 = PortfolioVectorMemory(2, 3)
    # remember: PVM's memory contains capacity + 1 item, because it also contains
    # the initial standard action of the agent (1, 0, 0, 0, 0, 0)...
    for i in range(4):
        pvm_1.add_at(np.ones(4) * i, i)
        if i < 3:
            pvm_2.add_at(np.ones(4) * i, i)
    assert (
        np.testing.assert_array_equal(
            np.array(pvm_1.memory),
            np.array([np.ones(4) * 0, np.ones(4) * 1, np.ones(4) * 2, np.ones(4) * 3]),
        )
        is None
    )
    assert (
        np.testing.assert_array_equal(
            np.array(pvm_2.memory),
            np.array([np.ones(4) * 0, np.ones(4) * 1, np.ones(4) * 2]),
        )
        is None
    )
    new_pvm = combine_portfolio_vector_memories([pvm_1, pvm_2])
    assert (
        np.testing.assert_array_equal(
            np.array(new_pvm.memory),
            np.array(
                [
                    np.ones(4) * 0,
                    np.ones(4) * 1,
                    np.ones(4) * 2,
                    np.ones(4) * 3,
                    np.ones(4) * 1,
                    np.ones(4) * 2,
                ]
            ),
        )
        is None
    )
    assert new_pvm.capacity == 5
    assert new_pvm.index == 3
    new_pvm = combine_portfolio_vector_memories([pvm_1, pvm_2], move_index=False)
    assert (
        np.testing.assert_array_equal(
            np.array(new_pvm.memory),
            np.array(
                [
                    np.ones(4) * 0,
                    np.ones(4) * 1,
                    np.ones(4) * 2,
                    np.ones(4) * 3,
                    np.ones(4) * 1,
                    np.ones(4) * 2,
                ]
            ),
        )
        is None
    )
    assert new_pvm.capacity == 5
    assert new_pvm.index == 0


def test_combine_multiple_pvms():
    """Tests if multiple portfolio vector memories are being correctly combined."""
    pvm_1 = PortfolioVectorMemory(3, 3)
    pvm_2 = PortfolioVectorMemory(2, 3)
    pvm_3 = PortfolioVectorMemory(1, 3)
    pvm_4 = PortfolioVectorMemory(3, 3)
    # remember: PVM's memory contains capacity + 1 item, because it also contains
    # the initial standard action of the agent (1, 0, 0, 0, 0, 0)...
    for i in range(4):
        pvm_1.add_at(np.ones(4) * i, i)
        pvm_4.add_at(np.ones(4) * i, i)
        if i < 3:
            pvm_2.add_at(np.ones(4) * i, i)
        if i < 2:
            pvm_3.add_at(np.ones(4) * i, i)
    new_pvm = combine_portfolio_vector_memories([pvm_1, pvm_2, pvm_3, pvm_4])
    assert (
        np.testing.assert_array_equal(
            np.array(new_pvm.memory),
            np.array(
                [
                    np.ones(4) * 0,
                    np.ones(4) * 1,
                    np.ones(4) * 2,
                    np.ones(4) * 3,
                    np.ones(4) * 1,
                    np.ones(4) * 2,
                    np.ones(4) * 1,
                    np.ones(4) * 1,
                    np.ones(4) * 2,
                    np.ones(4) * 3,
                ]
            ),
        )
        is None
    )
    assert new_pvm.capacity == 9
    assert new_pvm.index == 6


def test_combine_pvms_asserterror():
    """Tests if an assertion error will be raised when pvms of different portfolio
    sizes are
    """
    pvm_1 = PortfolioVectorMemory(3, 3)
    pvm_2 = PortfolioVectorMemory(2, 2)
    with pytest.raises(AssertionError):
        combine_portfolio_vector_memories([pvm_1, pvm_2])
