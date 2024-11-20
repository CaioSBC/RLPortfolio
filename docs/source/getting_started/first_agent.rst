Your First Agent
================

Fist, It is necessary to instantiate an environment object. The environment makes use of a `Pandas <https://pandas.pydata.org/>`_ dataframe which contains the time series of price of stocks.

.. code-block:: python

    import pandas as pd
    from rlportfolio.environment import PortfolioOptimizationEnv

    # dataframe with training data (market price time series)
    df_train = pd.read_csv("train_data.csv")

    environment = PortfolioOptimizationEnv(
            df_train, # data to be used
            100000    # initial value of the portfolio
        )

Now, we instantiate the policy gradient algorithm to generate an agent that actuates in the created environment. Note that, in this step, we also define the policy of actions of the agent.

.. code-block:: python
    
    from rlportfolio.algorithm import PolicyGradient
    from rlportfolio.policy import EI3

    algorithm = PolicyGradient(environment, policy=EI3)

Finally, the algorithm is used to train the agent for 10000 training steps.

.. code-block:: python

    # train the algorithm for 10000 steps
    algorithm.train(10000)

The algorithm also contains an interface which allows the agent to be tested in new market data.

.. code-block:: python

    # dataframe with testing data (market price time series)
    df_test = pd.read_csv("test_data.csv")

    environment_test = PortfolioOptimizationEnv(
            df_test, # data to be used
            100000   # initial value of the portfolio
        )

    # test the agent in the test environment
    algorithm.test(environment_test)

The test function will return a Python dictionary containing several performance metrics of the agent.
