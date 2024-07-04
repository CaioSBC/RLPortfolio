from __future__ import annotations

import pandas as pd
import numpy as np
import torch

from rl_portfolio.algorithm import PolicyGradient
from rl_portfolio.environment import PortfolioOptimizationEnv

# dataframe with fake data to use in the tests
test_dataframe = pd.DataFrame(
    {
        "tic": [
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "C",
        ],
        "date": [
            "2024-04-22",
            "2024-04-23",
            "2024-04-24",
            "2024-04-25",
            "2024-04-26",
            "2024-04-22",
            "2024-04-23",
            "2024-04-24",
            "2024-04-25",
            "2024-04-26",
            "2024-04-22",
            "2024-04-23",
            "2024-04-24",
            "2024-04-25",
            "2024-04-26",
        ],
        "feature_1": [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            2.0,
            1.0,
            0.5,
            0.25,
            0.5,
            5.0,
            2.5,
            5.0,
            2.5,
            1.25,
        ],
        "feature_2": [
            1.5,
            0.75,
            0.25,
            1.0,
            2.0,
            2.0,
            1.0,
            0.5,
            1.5,
            3.0,
            1.0,
            0.5,
            2.0,
            1.0,
            3.0,
        ],
    },
)

# environment using test dataframe
environment = PortfolioOptimizationEnv(
    test_dataframe,
    1000,
    features=["feature_1", "feature_2"],
    valuation_feature="feature_1",
    time_format="%Y-%m-%d",
    time_window=2,
    print_metrics=False,
    plot_graphs=False,
)


def test_buffers_after_episode():
    """Tests if the replay buffer and portfolio vector memories are correctly saving
    experiences after an episode."""
    algorithm = PolicyGradient(
        environment,
        policy_kwargs={"initial_features": 2, "k_size": 1, "time_window": 2},
    )
    algorithm._run_episode()
    # PVM
    assert len(algorithm.train_pvm) == 4
    np.testing.assert_allclose(algorithm.train_pvm.memory[0], np.array([1, 0, 0, 0]))

    # REPLAY BUFFER
    assert len(algorithm.train_buffer) == 3
    # check first item
    obs, last_action, price_variation, index = algorithm.train_buffer.buffer[0]
    expected_obs = np.array(
        [
            [[1.0, 2.0], [2.0, 1.0], [5.0, 2.5]],
            [[1.5, 0.75], [2.0, 1.0], [1.0, 0.5]],
        ]
    )
    expected_last_action = np.array([1, 0, 0, 0])
    expected_price_variation = np.array([1, 3 / 2, 0.5, 2.0])
    expected_index = 0
    np.testing.assert_allclose(obs, expected_obs)
    np.testing.assert_allclose(last_action, expected_last_action)
    np.testing.assert_allclose(price_variation, expected_price_variation)
    assert index == expected_index
    # check second item
    obs, last_action, price_variation, index = algorithm.train_buffer.buffer[1]
    expected_obs = np.array(
        [
            [[2.0, 3.0], [1.0, 0.5], [2.5, 5.0]],
            [[0.75, 0.25], [1.0, 0.5], [0.5, 2.0]],
        ]
    )
    expected_price_variation = np.array([1, 4 / 3, 0.5, 0.5])
    expected_index = 1
    np.testing.assert_allclose(obs, expected_obs)
    np.testing.assert_allclose(price_variation, expected_price_variation)
    assert index == expected_index
    # check third item
    obs, last_action, price_variation, index = algorithm.train_buffer.buffer[2]
    expected_obs = np.array(
        [
            [[3.0, 4.0], [0.5, 0.25], [5.0, 2.5]],
            [[0.25, 1.0], [0.5, 1.5], [2.0, 1.0]],
        ]
    )

    expected_price_variation = np.array([1, 5 / 4, 2.0, 0.5])
    expected_index = 2
    np.testing.assert_allclose(obs, expected_obs)
    np.testing.assert_allclose(price_variation, expected_price_variation)
    assert index == expected_index


def test_buffers_update():
    """Tests if replay buffer and portfolio vector memory are correctly updated."""
    algorithm = PolicyGradient(
        environment,
        policy_kwargs={"initial_features": 2, "k_size": 1, "time_window": 2},
    )
    algorithm._run_episode()
    # Updates items
    actions = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0]])
    indexes = torch.tensor([0, 1])
    algorithm._update_buffers(
        actions, indexes, test=False, update_rb=True, update_pvm=True
    )
    # PVM
    assert len(algorithm.train_pvm) == 4
    np.testing.assert_allclose(algorithm.train_pvm.memory[0], np.array([1, 0, 0, 0]))
    np.testing.assert_allclose(algorithm.train_pvm.memory[1], np.array([0, 1, 0, 0]))
    np.testing.assert_allclose(algorithm.train_pvm.memory[2], np.array([0, 0, 1, 0]))

    # REPLAY BUFFER
    # check first item
    obs, last_action, price_variation, index = algorithm.train_buffer.buffer[0]
    expected_obs = np.array(
        [
            [[1.0, 2.0], [2.0, 1.0], [5.0, 2.5]],
            [[1.5, 0.75], [2.0, 1.0], [1.0, 0.5]],
        ]
    )
    expected_last_action = np.array([1, 0, 0, 0])
    expected_price_variation = np.array([1, 3 / 2, 0.5, 2.0])
    expected_index = 0
    np.testing.assert_allclose(obs, expected_obs)
    np.testing.assert_allclose(last_action, expected_last_action)
    np.testing.assert_allclose(price_variation, expected_price_variation)
    assert index == expected_index
    # check second item
    obs, last_action, price_variation, index = algorithm.train_buffer.buffer[1]
    expected_obs = np.array(
        [
            [[2.0, 3.0], [1.0, 0.5], [2.5, 5.0]],
            [[0.75, 0.25], [1.0, 0.5], [0.5, 2.0]],
        ]
    )
    expected_last_action = np.array([0, 1, 0, 0])
    expected_price_variation = np.array([1, 4 / 3, 0.5, 0.5])
    expected_index = 1
    np.testing.assert_allclose(obs, expected_obs)
    np.testing.assert_allclose(price_variation, expected_price_variation)
    assert index == expected_index
    # check third item
    obs, last_action, price_variation, index = algorithm.train_buffer.buffer[2]
    expected_obs = np.array(
        [
            [[3.0, 4.0], [0.5, 0.25], [5.0, 2.5]],
            [[0.25, 1.0], [0.5, 1.5], [2.0, 1.0]],
        ]
    )
    expected_last_action = np.array([0, 0, 1, 0])
    expected_price_variation = np.array([1, 5 / 4, 2.0, 0.5])
    expected_index = 2
    np.testing.assert_allclose(obs, expected_obs)
    np.testing.assert_allclose(price_variation, expected_price_variation)
    assert index == expected_index


def test_replay_buffer_update_last_index():
    """Tests if the replay buffer is correctly updated when the experience of the
    buffer (consequently, the last index) is chosen in the batch.
    """
    algorithm = PolicyGradient(
        environment,
        policy_kwargs={"initial_features": 2, "k_size": 1, "time_window": 2},
    )
    algorithm._run_episode()
    # Updates one item
    actions = torch.tensor([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
    indexes = torch.tensor([0, 1, 2])
    algorithm._update_buffers(
        actions, indexes, test=False, update_rb=True, update_pvm=True
    )
    # PVM
    assert len(algorithm.train_pvm) == 4
    np.testing.assert_allclose(algorithm.train_pvm.memory[0], np.array([1, 0, 0, 0]))
    np.testing.assert_allclose(algorithm.train_pvm.memory[1], np.array([0, 0, 0, 1]))
    np.testing.assert_allclose(algorithm.train_pvm.memory[2], np.array([0, 1, 0, 0]))
    np.testing.assert_allclose(algorithm.train_pvm.memory[3], np.array([0, 0, 1, 0]))

    # REPLAY BUFFER
    # check first item
    obs, last_action, price_variation, index = algorithm.train_buffer.buffer[0]
    expected_obs = np.array(
        [
            [[1.0, 2.0], [2.0, 1.0], [5.0, 2.5]],
            [[1.5, 0.75], [2.0, 1.0], [1.0, 0.5]],
        ]
    )
    expected_last_action = np.array([1, 0, 0, 0])
    expected_price_variation = np.array([1, 3 / 2, 0.5, 2.0])
    expected_index = 0
    np.testing.assert_allclose(obs, expected_obs)
    np.testing.assert_allclose(last_action, expected_last_action)
    np.testing.assert_allclose(price_variation, expected_price_variation)
    assert index == expected_index
    # check second item
    obs, last_action, price_variation, index = algorithm.train_buffer.buffer[1]
    expected_obs = np.array(
        [
            [[2.0, 3.0], [1.0, 0.5], [2.5, 5.0]],
            [[0.75, 0.25], [1.0, 0.5], [0.5, 2.0]],
        ]
    )
    expected_last_action = np.array([0, 0, 0, 1])
    expected_price_variation = np.array([1, 4 / 3, 0.5, 0.5])
    expected_index = 1
    np.testing.assert_allclose(obs, expected_obs)
    np.testing.assert_allclose(price_variation, expected_price_variation)
    assert index == expected_index
    # check third item
    obs, last_action, price_variation, index = algorithm.train_buffer.buffer[2]
    expected_obs = np.array(
        [
            [[3.0, 4.0], [0.5, 0.25], [5.0, 2.5]],
            [[0.25, 1.0], [0.5, 1.5], [2.0, 1.0]],
        ]
    )
    expected_last_action = np.array([0, 1, 0, 0])
    expected_price_variation = np.array([1, 5 / 4, 2.0, 0.5])
    expected_index = 2
    np.testing.assert_allclose(obs, expected_obs)
    np.testing.assert_allclose(price_variation, expected_price_variation)
    assert index == expected_index
