from __future__ import annotations

import pandas as pd
import numpy as np

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
    time_window=3,
    print_metrics=False,
    plot_graphs=False,
)


def test_replay_buffer_after_episode():
    algorithm = PolicyGradient(
        environment,
        policy_kwargs={"initial_features": 2, "k_size": 2, "time_window": 3},
    )
    algorithm._run_episode()
    assert len(algorithm.train_buffer) == 2
    # check first item
    obs, last_action, price_variation, index = algorithm.train_buffer.buffer[0]
    expected_obs = np.array(
        [
            [[1.0, 2.0, 3.0], [2.0, 1.0, 0.5], [5.0, 2.5, 5.0]],
            [[1.5, 0.75, 0.25], [2.0, 1.0, 0.5], [1.0, 0.5, 2.0]],
        ]
    )
    expected_last_action = np.array([1, 0, 0, 0])
    expected_price_variation = np.array([1, 4 / 3, 0.5, 0.5])
    expected_index = 0
    np.testing.assert_allclose(obs, expected_obs)
    np.testing.assert_allclose(last_action, expected_last_action)
    np.testing.assert_allclose(price_variation, expected_price_variation)
    assert index == expected_index
    # check second item
    obs, last_action, price_variation, index = algorithm.train_buffer.buffer[1]
    expected_obs = np.array(
        [
            [[2.0, 3.0, 4.0], [1.0, 0.5, 0.25], [2.5, 5.0, 2.5]],
            [[0.75, 0.25, 1.0], [1.0, 0.5, 1.5], [0.5, 2.0, 1.0]],
        ]
    )
    
    expected_price_variation = np.array([1, 5 / 4, 2.0, 0.5])
    expected_index = 1
    np.testing.assert_allclose(obs, expected_obs)
    np.testing.assert_allclose(price_variation, expected_price_variation)
    assert index == expected_index

