Getting Starting
================

Components of RLPortfolio
-------------------------

Before utilizing the library, it is important to understand that it consists of three separate components that work together in order to execute the reinforcement learning process of the agent:

* The **Environment** is a simulation of the market based on historical price time series. The agent performs actions in this environment (i.e. the percentage of value invested in each stock of the portfolio) and it calculates the effects of the passage of time.
* The **Algorithm** is the training algorithm. It applies the mathematical formulations of the reinforcement learning process that allow the agent to learn an optimal policy of actions.
* The **Policy** is the agent's policy of actions, which defines how the agent will act given the current state of the market.

Now, it is possible to combine these components to train your first agent.

Training your First Agent
-------------------------

Fist, It is necessary to instantiate an environment object. The environment makes use of a dataframe which contains the time series of price of stocks.

.. code-block:: python

    import pandas as pd
    from rlportfolio.environment import PortfolioOptimizationEnv

    # dataframe with training data (market price time series)
    df_train = pd.read_csv("train_data.csv")

    environment = PortfolioOptimizationEnv(
            df_train, # data to be used
            100000    # initial value of the portfolio
        )

Now, we instantiate the policy gradient algorithm to generate an agent that actuates in the created environment.


.. code-block:: python
    
    from rlportfolio.algorithm import PolicyGradient
    from rlportfolio.policy import EI3

    algorithm = PolicyGradient(environment, policy=EI3)

