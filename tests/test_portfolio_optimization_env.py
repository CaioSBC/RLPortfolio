from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datetime import datetime

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

# environment with normal observation
environment = PortfolioOptimizationEnv(
    test_dataframe,
    1000,
    normalize_df=None,
    features=["feature_1", "feature_2"],
    valuation_feature="feature_1",
    time_window=3,
)

# environment with dict observation
environment_dict = PortfolioOptimizationEnv(
    test_dataframe,
    1000,
    normalize_df=None,
    features=["feature_1", "feature_2"],
    valuation_feature="feature_1",
    time_window=3,
    return_last_action=True,
)

# list of portfolio values
portfolio_values = [1000]


def test_environment_properties():
    """Tests the main properties of the environment."""
    assert environment.portfolio_size == 3
    assert environment.episode_length == 3
    assert environment.action_space.shape == (4,)
    assert environment.observation_space.shape == (2, 3, 3)


def test_environment_reset():
    """Tests the content of observations and info returned in reset()."""
    obs, info = environment.reset()
    expected_obs = np.array(
        [
            [[1.0, 2.0, 3.0], [2.0, 1.0, 0.5], [5.0, 2.5, 5.0]],
            [[1.5, 0.75, 0.25], [2.0, 1.0, 0.5], [1.0, 0.5, 2.0]],
        ]
    )

    expected_data = test_dataframe[
        (test_dataframe["date"] >= "2024-04-22")
        & (test_dataframe["date"] <= "2024-04-24")
    ][["date", "tic", "feature_1", "feature_2"]]
    expected_data["date"] = pd.to_datetime(expected_data["date"])
    expected_data["feature_1"] = expected_data["feature_1"].astype("float32")
    expected_data["feature_2"] = expected_data["feature_2"].astype("float32")

    expected_price_variation = np.array([1, 3 / 2, 0.5, 2.0])

    assert environment._portfolio_value == 1000
    assert obs.shape == (2, 3, 3)
    assert pytest.approx(obs) == expected_obs
    assert info["tics"].tolist() == ["A", "B", "C"]
    assert info["start_time"] == datetime.strptime("2024-04-22", "%Y-%m-%d")
    assert info["start_time_index"] == 0
    assert info["end_time"] == datetime.strptime("2024-04-24", "%Y-%m-%d")
    assert info["end_time_index"] == 2
    assert pd.testing.assert_frame_equal(info["data"], expected_data) is None
    assert (
        np.testing.assert_array_almost_equal(
            info["price_variation"], expected_price_variation
        )
        is None
    )


def test_environment_step_1():
    """Tests the content of the values returned in step()."""
    action = np.array([0.25, 0.15, 0.35, 0.25])
    obs, reward, terminal, truncated, info = environment.step(action)
    expected_obs = np.array(
        [
            [[2.0, 3.0, 4.0], [1.0, 0.5, 0.25], [2.5, 5.0, 2.5]],
            [[0.75, 0.25, 1.0], [1.0, 0.5, 1.5], [0.5, 2.0, 1.0]],
        ]
    )

    expected_data = test_dataframe[
        (test_dataframe["date"] >= "2024-04-23")
        & (test_dataframe["date"] <= "2024-04-25")
    ][["date", "tic", "feature_1", "feature_2"]]
    expected_data["date"] = pd.to_datetime(expected_data["date"])
    expected_data["feature_1"] = expected_data["feature_1"].astype("float32")
    expected_data["feature_2"] = expected_data["feature_2"].astype("float32")

    expected_price_variation = np.array([1, 4 / 3, 0.5, 0.5])

    assert pytest.approx(environment._portfolio_value) == pytest.approx(
        np.sum(portfolio_values[-1] * (action * expected_price_variation))
    )
    portfolio_values.append(environment._portfolio_value)

    assert obs.shape == (2, 3, 3)
    assert pytest.approx(obs) == expected_obs
    assert pytest.approx(reward) == pytest.approx(
        np.log(portfolio_values[-1] / portfolio_values[-2])
    )
    assert terminal == False
    assert truncated == False
    assert info["tics"].tolist() == ["A", "B", "C"]
    assert info["start_time"] == datetime.strptime("2024-04-23", "%Y-%m-%d")
    assert info["start_time_index"] == 1
    assert info["end_time"] == datetime.strptime("2024-04-25", "%Y-%m-%d")
    assert info["end_time_index"] == 3
    assert pd.testing.assert_frame_equal(info["data"], expected_data) is None
    assert (
        np.testing.assert_array_almost_equal(
            info["price_variation"], expected_price_variation
        )
        is None
    )


def test_environment_step_2():
    """Tests the content of the values returned in step()."""
    action = np.array([0.10, 0.50, 0.05, 0.35])
    obs, reward, terminal, truncated, info = environment.step(action)
    expected_obs = np.array(
        [
            [[3.0, 4.0, 5.0], [0.5, 0.25, 0.5], [5.0, 2.5, 1.25]],
            [[0.25, 1.0, 2.0], [0.5, 1.5, 3.0], [2.0, 1.0, 3.0]],
        ]
    )

    expected_data = test_dataframe[
        (test_dataframe["date"] >= "2024-04-24")
        & (test_dataframe["date"] <= "2024-04-26")
    ][["date", "tic", "feature_1", "feature_2"]]
    expected_data["date"] = pd.to_datetime(expected_data["date"])
    expected_data["feature_1"] = expected_data["feature_1"].astype("float32")
    expected_data["feature_2"] = expected_data["feature_2"].astype("float32")

    expected_price_variation = np.array([1, 5 / 4, 2.0, 0.5])

    assert pytest.approx(environment._portfolio_value) == pytest.approx(
        np.sum(portfolio_values[-1] * (action * expected_price_variation))
    )
    portfolio_values.append(environment._portfolio_value)

    assert obs.shape == (2, 3, 3)
    assert pytest.approx(obs) == expected_obs
    assert pytest.approx(reward) == pytest.approx(
        np.log(portfolio_values[-1] / portfolio_values[-2])
    )
    assert terminal == False
    assert truncated == False
    assert info["tics"].tolist() == ["A", "B", "C"]
    assert info["start_time"] == datetime.strptime("2024-04-24", "%Y-%m-%d")
    assert info["start_time_index"] == 2
    assert info["end_time"] == datetime.strptime("2024-04-26", "%Y-%m-%d")
    assert info["end_time_index"] == 4
    assert pd.testing.assert_frame_equal(info["data"], expected_data) is None
    assert (
        np.testing.assert_array_almost_equal(
            info["price_variation"], expected_price_variation
        )
        is None
    )


def test_environment_step_3():
    """Tests the content of the values returned in step(). Note that, in
    this case, we have a terminal state.
    """
    action = np.array([0.00, 0.20, 0.10, 0.70])
    obs, reward, terminal, truncated, info = environment.step(action)
    expected_obs = np.array(
        [
            [[3.0, 4.0, 5.0], [0.5, 0.25, 0.5], [5.0, 2.5, 1.25]],
            [[0.25, 1.0, 2.0], [0.5, 1.5, 3.0], [2.0, 1.0, 3.0]],
        ]
    )

    expected_data = test_dataframe[
        (test_dataframe["date"] >= "2024-04-24")
        & (test_dataframe["date"] <= "2024-04-26")
    ][["date", "tic", "feature_1", "feature_2"]]
    expected_data["date"] = pd.to_datetime(expected_data["date"])
    expected_data["feature_1"] = expected_data["feature_1"].astype("float32")
    expected_data["feature_2"] = expected_data["feature_2"].astype("float32")

    expected_price_variation = np.array([1, 5 / 4, 2.0, 0.5])

    assert pytest.approx(environment._portfolio_value) == pytest.approx(
        portfolio_values[-1]
    )

    assert obs.shape == (2, 3, 3)
    assert pytest.approx(obs) == expected_obs
    assert pytest.approx(reward) == pytest.approx(
        np.log(portfolio_values[-1] / portfolio_values[-2])
    )
    assert terminal == True
    assert truncated == False
    assert info["tics"].tolist() == ["A", "B", "C"]
    assert info["start_time"] == datetime.strptime("2024-04-24", "%Y-%m-%d")
    assert info["start_time_index"] == 2
    assert info["end_time"] == datetime.strptime("2024-04-26", "%Y-%m-%d")
    assert info["end_time_index"] == 4
    assert pd.testing.assert_frame_equal(info["data"], expected_data) is None
    assert (
        np.testing.assert_array_almost_equal(
            info["price_variation"], expected_price_variation
        )
        is None
    )
    # reset portfolio value list
    portfolio_values.clear()


def test_environment_dict_properties():
    """Tests the main properties of the dict-observation environment."""
    assert environment_dict.portfolio_size == 3
    assert environment_dict.episode_length == 3
    assert environment_dict.action_space.shape == (4,)
    assert environment_dict.observation_space["state"].shape == (2, 3, 3)
    assert environment_dict.observation_space["last_action"].shape == (4,)


def test_environment_dict_observations():
    """Tests the consistency of observations returned by the environment
    during multiple step() calls."""
    obs, _ = environment_dict.reset()
    expected_obs = np.array(
        [
            [[1.0, 2.0, 3.0], [2.0, 1.0, 0.5], [5.0, 2.5, 5.0]],
            [[1.5, 0.75, 0.25], [2.0, 1.0, 0.5], [1.0, 0.5, 2.0]],
        ]
    )
    assert type(obs) == dict
    assert pytest.approx(obs["state"]) == expected_obs
    assert pytest.approx(obs["last_action"]) == np.array([1, 0, 0, 0])

    obs, _, _, _, _ = environment_dict.step([0.2, 0.5, 0.1, 0.2])
    expected_obs = np.array(
        [
            [[2.0, 3.0, 4.0], [1.0, 0.5, 0.25], [2.5, 5.0, 2.5]],
            [[0.75, 0.25, 1.0], [1.0, 0.5, 1.5], [0.5, 2.0, 1.0]],
        ]
    )
    assert type(obs) == dict
    assert pytest.approx(obs["state"]) == expected_obs
    assert pytest.approx(obs["last_action"]) == np.array([0.2, 0.5, 0.1, 0.2])

    obs, _, _, _, _ = environment_dict.step([0.4, 0.1, 0.3, 0.2])
    expected_obs = np.array(
        [
            [[3.0, 4.0, 5.0], [0.5, 0.25, 0.5], [5.0, 2.5, 1.25]],
            [[0.25, 1.0, 2.0], [0.5, 1.5, 3.0], [2.0, 1.0, 3.0]],
        ]
    )
    assert type(obs) == dict
    assert pytest.approx(obs["state"]) == expected_obs
    assert pytest.approx(obs["last_action"]) == np.array([0.4, 0.1, 0.3, 0.2])

    obs, _, _, _, _ = environment_dict.step([0.1, 0.1, 0.6, 0.2])
    expected_obs = np.array(
        [
            [[3.0, 4.0, 5.0], [0.5, 0.25, 0.5], [5.0, 2.5, 1.25]],
            [[0.25, 1.0, 2.0], [0.5, 1.5, 3.0], [2.0, 1.0, 3.0]],
        ]
    )
    assert type(obs) == dict
    assert pytest.approx(obs["state"]) == expected_obs
    assert pytest.approx(obs["last_action"]) == np.array([0.4, 0.1, 0.3, 0.2])
