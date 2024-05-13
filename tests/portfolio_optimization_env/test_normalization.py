from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datetime import datetime

from rl_portfolio.environment import PortfolioOptimizationEnv
from rl_portfolio.data import GroupByScaler

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


def test_by_previous_time_norm():
    env = PortfolioOptimizationEnv(
        test_dataframe,
        1000,
        data_normalization="by_previous_time",
        features=["feature_1", "feature_2"],
        valuation_feature="feature_1",
        time_format="%Y-%m-%d",
        time_window=3,
        print_metrics=False,
        plot_graphs=False,
    )

    assert env._df["tic"].to_list() == [
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
    ]
    assert env._df["date"].dt.strftime("%Y-%m-%d").to_list() == [
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
    ]
    assert env._df["feature_1"].to_list() == pytest.approx(
        [1.0, 2.0, 1.5, 4 / 3, 1.25, 1.0, 0.5, 0.5, 0.5, 2.0, 1.0, 0.5, 2.0, 0.5, 0.5]
    )
    assert env._df["feature_2"].to_list() == pytest.approx(
        [1.0, 0.5, 1 / 3, 4.0, 2.0, 1.0, 0.5, 0.5, 3.0, 2.0, 1.0, 0.5, 4.0, 0.5, 3.0]
    )


def test_by_column_norm():
    env = PortfolioOptimizationEnv(
        test_dataframe,
        1000,
        data_normalization="by_feature_2",
        features=["feature_1", "feature_2"],
        valuation_feature="feature_1",
        time_format="%Y-%m-%d",
        time_window=3,
        print_metrics=False,
        plot_graphs=False,
    )
    assert env._df["tic"].to_list() == [
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
    ]
    assert env._df["date"].dt.strftime("%Y-%m-%d").to_list() == [
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
    ]
    assert env._df["feature_1"].to_list() == pytest.approx(
        [
            2 / 3,
            8 / 3,
            12.0,
            4.0,
            5 / 2,
            1.0,
            1.0,
            1.0,
            1 / 6,
            1 / 6,
            5.0,
            5.0,
            5 / 2,
            2.5,
            5 / 12,
        ]
    )
    assert env._df["feature_2"].to_list() == pytest.approx(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )


def test_callable_dataframe_norm():
    def callable_func(dataframe):
        new_dataframe = dataframe.copy()
        new_dataframe["feature_1"] = dataframe["feature_1"] / 10
        return new_dataframe

    env = PortfolioOptimizationEnv(
        test_dataframe,
        1000,
        data_normalization=callable_func,
        features=["feature_1", "feature_2"],
        valuation_feature="feature_1",
        time_format="%Y-%m-%d",
        time_window=3,
        print_metrics=False,
        plot_graphs=False,
    )

    assert env._df["tic"].to_list() == [
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
    ]
    assert env._df["date"].dt.strftime("%Y-%m-%d").to_list() == [
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
    ]
    assert env._df["feature_1"].to_list() == pytest.approx(
        [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.2,
            0.1,
            0.05,
            0.025,
            0.05,
            0.5,
            0.25,
            0.5,
            0.25,
            0.125,
        ]
    )
    assert env._df["feature_2"].to_list() == pytest.approx(
        [
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
        ]
    )


def test_groupby_scaler_norm():
    scaler = GroupByScaler(by="tic", columns=["feature_1", "feature_2"])

    env = PortfolioOptimizationEnv(
        test_dataframe,
        1000,
        data_normalization=scaler.fit_transform,
        features=["feature_1", "feature_2"],
        valuation_feature="feature_1",
        time_format="%Y-%m-%d",
        time_window=3,
        print_metrics=False,
        plot_graphs=False,
    )

    assert env._df["tic"].to_list() == [
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
    ]
    assert env._df["date"].dt.strftime("%Y-%m-%d").to_list() == [
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
    ]
    assert env._df["feature_1"].to_list() == pytest.approx(
        [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 0.5, 0.25, 0.125, 0.25, 1.0, 0.5, 1.0, 0.5, 0.25]
    )
    assert env._df["feature_2"].to_list() == pytest.approx(
        [
            0.75,
            0.375,
            0.125,
            0.5,
            1.0,
            2 / 3,
            1 / 3,
            1 / 6,
            0.5,
            1.0,
            1 / 3,
            1 / 6,
            2 / 3,
            1 / 3,
            1.0,
        ]
    )

    env = PortfolioOptimizationEnv(
        test_dataframe,
        1000,
        data_normalization=scaler.transform,
        features=["feature_1", "feature_2"],
        valuation_feature="feature_1",
        time_format="%Y-%m-%d",
        time_window=3,
        print_metrics=False,
        plot_graphs=False,
    )

    assert env._df["tic"].to_list() == [
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
    ]
    assert env._df["date"].dt.strftime("%Y-%m-%d").to_list() == [
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
    ]
    assert env._df["feature_1"].to_list() == pytest.approx(
        [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 0.5, 0.25, 0.125, 0.25, 1.0, 0.5, 1.0, 0.5, 0.25]
    )
    assert env._df["feature_2"].to_list() == pytest.approx(
        [
            0.75,
            0.375,
            0.125,
            0.5,
            1.0,
            2 / 3,
            1 / 3,
            1 / 6,
            0.5,
            1.0,
            1 / 3,
            1 / 6,
            2 / 3,
            1 / 3,
            1.0,
        ]
    )


def test_by_initial_value_state_norm():
    env = PortfolioOptimizationEnv(
        test_dataframe,
        1000,
        state_normalization="by_initial_value",
        features=["feature_1", "feature_2"],
        valuation_feature="feature_1",
        time_format="%Y-%m-%d",
        time_window=3,
        print_metrics=False,
        plot_graphs=False,
    )

    obs, _ = env.reset()
    expected_obs = np.array(
        [
            [[1.0, 2.0, 3.0], [1.0, 0.5, 0.25], [1.0, 0.5, 1.0]],
            [[1.0, 0.5, 1 / 6], [1.0, 0.5, 0.25], [1.0, 0.5, 2.0]],
        ]
    )
    assert pytest.approx(obs) == expected_obs

    obs, _, _, _, _ = env.step(np.array([1, 0, 0, 0]))
    expected_obs = np.array(
        [
            [[1.0, 1.5, 2.0], [1.0, 0.5, 0.25], [1.0, 2.0, 1.0]],
            [[1.0, 1 / 3, 4 / 3], [1.0, 0.5, 1.5], [1.0, 4.0, 2.0]],
        ]
    )
    assert pytest.approx(obs) == expected_obs

    obs, _, _, _, _ = env.step(np.array([1, 0, 0, 0]))
    expected_obs = np.array(
        [
            [[1.0, 4 / 3, 5 / 3], [1.0, 0.5, 1.0], [1.0, 0.5, 0.25]],
            [[1.0, 4.0, 8.0], [1.0, 3.0, 6.0], [1.0, 0.5, 1.5]],
        ]
    )
    assert pytest.approx(obs) == expected_obs


def test_by_last_value_state_norm():
    env = PortfolioOptimizationEnv(
        test_dataframe,
        1000,
        state_normalization="by_last_value",
        features=["feature_1", "feature_2"],
        valuation_feature="feature_1",
        time_format="%Y-%m-%d",
        time_window=3,
        print_metrics=False,
        plot_graphs=False,
    )

    obs, _ = env.reset()
    expected_obs = np.array(
        [
            [[1 / 3, 2 / 3, 1.0], [4.0, 2.0, 1.0], [1.0, 0.5, 1.0]],
            [[6.0, 3.0, 1.0], [4.0, 2.0, 1.0], [0.5, 0.25, 1.0]],
        ]
    )
    assert pytest.approx(obs) == expected_obs

    obs, _, _, _, _ = env.step(np.array([1, 0, 0, 0]))
    expected_obs = np.array(
        [
            [[0.5, 0.75, 1.0], [4.0, 2.0, 1.0], [1.0, 2.0, 1.0]],
            [[0.75, 0.25, 1.0], [2 / 3, 1 / 3, 1.0], [0.5, 2.0, 1.0]],
        ]
    )
    assert pytest.approx(obs) == expected_obs

    obs, _, _, _, _ = env.step(np.array([1, 0, 0, 0]))
    expected_obs = np.array(
        [
            [[0.6, 0.8, 1.0], [1.0, 0.5, 1.0], [4.0, 2.0, 1.0]],
            [[0.125, 0.5, 1.0], [1 / 6, 0.5, 1.0], [2 / 3, 1 / 3, 1.0]],
        ]
    )
    assert pytest.approx(obs) == expected_obs


def test_by_initial_feature_value_state_norm():
    env = PortfolioOptimizationEnv(
        test_dataframe,
        1000,
        state_normalization="by_initial_feature_1",
        features=["feature_1", "feature_2"],
        valuation_feature="feature_1",
        time_format="%Y-%m-%d",
        time_window=3,
        print_metrics=False,
        plot_graphs=False,
    )

    obs, _ = env.reset()
    expected_obs = np.array(
        [
            [[1.0, 2.0, 3.0], [1.0, 0.5, 0.25], [1.0, 0.5, 1.0]],
            [[1.5, 0.75, 0.25], [1.0, 0.5, 0.25], [0.2, 0.1, 0.4]],
        ]
    )
    assert pytest.approx(obs) == expected_obs

    obs, _, _, _, _ = env.step(np.array([1, 0, 0, 0]))
    expected_obs = np.array(
        [
            [[1.0, 1.5, 2.0], [1.0, 0.5, 0.25], [1.0, 2.0, 1.0]],
            [[0.375, 0.125, 0.5], [1.0, 0.5, 1.5], [0.2, 0.8, 0.4]],
        ]
    )
    assert pytest.approx(obs) == expected_obs

    obs, _, _, _, _ = env.step(np.array([1, 0, 0, 0]))
    expected_obs = np.array(
        [
            [[1.0, 4 / 3, 5 / 3], [1.0, 0.5, 1.0], [1.0, 0.5, 0.25]],
            [[1 / 12, 1 / 3, 2 / 3], [1.0, 3.0, 6.0], [0.4, 0.2, 0.6]],
        ]
    )
    assert pytest.approx(obs) == expected_obs


def test_by_last_feature_value_state_norm():
    env = PortfolioOptimizationEnv(
        test_dataframe,
        1000,
        state_normalization="by_last_feature_2",
        features=["feature_1", "feature_2"],
        valuation_feature="feature_1",
        time_format="%Y-%m-%d",
        time_window=3,
        print_metrics=False,
        plot_graphs=False,
    )

    obs, _ = env.reset()
    expected_obs = np.array(
        [
            [[4.0, 8.0, 12.0], [4.0, 2.0, 1.0], [2.5, 1.25, 2.5]],
            [[6.0, 3.0, 1.0], [4.0, 2.0, 1.0], [0.5, 0.25, 1.0]],
        ]
    )
    assert pytest.approx(obs) == expected_obs

    obs, _, _, _, _ = env.step(np.array([1, 0, 0, 0]))
    expected_obs = np.array(
        [
            [[2.0, 3.0, 4.0], [2 / 3, 1 / 3, 1 / 6], [2.5, 5.0, 2.5]],
            [[0.75, 0.25, 1.0], [2 / 3, 1 / 3, 1.0], [0.5, 2.0, 1.0]],
        ]
    )
    assert pytest.approx(obs) == expected_obs

    obs, _, _, _, _ = env.step(np.array([1, 0, 0, 0]))
    expected_obs = np.array(
        [
            [[1.5, 2.0, 2.5], [1 / 6, 1 / 12, 1 / 6], [5 / 3, 5 / 6, 5 / 12]],
            [[0.125, 0.5, 1.0], [1 / 6, 0.5, 1.0], [2 / 3, 1 / 3, 1.0]],
        ]
    )
    assert pytest.approx(obs) == expected_obs
