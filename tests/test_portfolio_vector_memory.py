from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl_portfolio.algorithm.buffers import PortfolioVectorMemory
from rl_portfolio.environment import PortfolioOptimizationEnv

pvm = PortfolioVectorMemory(capacity=4, portfolio_size=5)

test_dataframe = pd.DataFrame(
    {
        "tic": [
            "A",
            "A",
            "A",
            "A",
        ],
        "date": [
            "2024-04-22",
            "2024-04-23",
            "2024-04-24",
            "2024-04-25",
        ],
        "feature_1": [
            1.0,
            2.0,
            3.0,
            4.0,
        ],
    },
)

# environment with dict observation
environment_dict = PortfolioOptimizationEnv(
    test_dataframe,
    1000,
    features=["feature_1"],
    valuation_feature="feature_1",
    time_window=1,
    return_last_action=True,
    print_metrics=False,
    plot_graphs=False,
)


def test_pvm_properties():
    """Tests vector memory initial properties."""
    assert len(pvm.memory) == 5
    assert (
        np.testing.assert_array_equal(
            pvm.memory, [np.array([1] + [0] * 5, dtype=np.float32)] * 5
        )
        is None
    )
    assert pvm.index == 0


def test_pvm_retrieve_add():
    """Tests the process of retrieving and adding values from pvm."""
    for i in range(20):
        assert pvm.index == i % 5
        value = pvm.retrieve()
        # in the first retrieve, returns initial action
        if i == 0:
            assert (
                np.testing.assert_array_equal(
                    value, np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)
                )
                is None
            )
        # in the other retrieves, returns the last action
        else:
            assert (
                np.testing.assert_array_equal(
                    value,
                    np.array(
                        [i - 1, i - 1, i - 1, i - 1, i - 1, i - 1], dtype=np.float32
                    ),
                )
                is None
            )
        pvm.add(np.array([i, i, i, i, i, i], dtype=np.float32))


def test_pvm_reset():
    """Tests the memory's reset process."""
    pvm.reset()
    assert len(pvm.memory) == 5
    assert (
        np.testing.assert_array_equal(
            pvm.memory, [np.array([1] + [0] * 5, dtype=np.float32)] * 5
        )
        is None
    )
    assert pvm.index == 0


def test_last_action():
    """Tests if memory's retrieved value is equal to environment's
    observation last action.
    """
    tmp_pvm = PortfolioVectorMemory(capacity=4, portfolio_size=1)
    obs, _ = environment_dict.reset()
    last_action = tmp_pvm.retrieve()
    assert np.testing.assert_array_almost_equal(obs["last_action"], last_action) is None
    done = False
    while not done:
        action = np.random.dirichlet(np.ones(2), size=1).astype(np.float32).squeeze()
        tmp_pvm.add(action)
        obs, _, terminal, truncated, _ = environment_dict.step(action)
        done = terminal or truncated
        if not done:
            last_action = tmp_pvm.retrieve()
            assert (
                np.testing.assert_array_almost_equal(obs["last_action"], last_action)
                is None
            )
