*PortfolioOptimizationEnv*
==========================

The *PortfolioOptimizationEnv* (or POE) is a `Gymnasium <https://gymnasium.farama.org/index.html>`_ environment implementing the mathematical formulation introduced in `A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem <https://doi.org/10.48550/arXiv.1706.10059>_` from Zhengyao Jiang and Dixing Xu and Jinjun Liang. 

Instantiating the environment
-----------------------------

Before using POE, it is necessary to provide the parameters it will use during the simulation. The most important one is the `Pandas <https://pandas.pydata.org/>`_ dataframe with historical data the environment will use to define the agent's observation.

For each asset in the portfolio, this dataframe must contain a date column containing the datetimes, a tic column with the name of the assets and feature columns which are considered as the temporal features to be considered. A valid dataframe would look like the following:

.. code-block:: python

                date        high            low             close           tic
        0   2020-12-23  0.157414        0.127420        0.136394        ADA-USD
        1   2020-12-23  34.381519       30.074295       31.097898       BNB-USD
        2   2020-12-23  24024.490234    22802.646484    23241.345703    BTC-USD
        3   2020-12-23  0.004735        0.003640        0.003768        DOGE-USD
        4   2020-12-23  637.122803      560.364258      583.714600      ETH-USD
        ... ...         ...             ...             ...             ...

In this example, the features are the high, low and close price time series. The name of the date column can be changed using the :code:`time_column` argument (by default, its value is "date") and the name of the tic columns can be set using the :code:`tic_column` parameter (by default, its "tic"). Finally, the list of features can be defined using the :code:`features` arguments: by default, its value is :code:`["close", "high", "low"]`. Any column that is not defined in these arguments is ignored.

.. note::

    For each datetime in the dataframe, there must be every feature for every tic in the portfolio. Thus, if there is no data related to a specific asset in a specific datetime, the environment will throw an error.

Additionaly, it is also necessary to specify the initial amount of cash that will be invested in the portfolio. With those two arguments, the environment can be instatiated:

.. code-block:: python

        import pandas as pd
        from rlportfolio.environment import PortfolioOptimizationEnv

         # loading example dataframe
        dataframe = pd.read_csv("my_dataset.csv")

        # instantiating environment using the example-dataframe and defining its
        # initial value as 100000
        env = PortfolioOptimizationEnv(dataframe, 100000)

As a reinforcement learning environment, POE is expected to receive the action of the agent as input and output the next observation and the reward achieved. 

The Action
----------

The action performed by the agent at time step :math:`t` is a vector :math:`\vec{W_{t}}` with the weights of each stock in the portfolio. In the research field, this vector is called *portfolio vector* or *weights vector* and, given a portfolio with :math:`n` assets, it contains :math:`n+1` values, because its first value :math:`\vec{W_{t}(0)}` relates to the weights of non-invested assets.

It is important to highlight that, to be valid, the portfolio vector must respect two constraints:

.. math::

    0 \le \vec{W_{t}}(i) \le 1,

    \sum\limits_{i=0}^{n} \vec{W_{t}}(i) = 1.

The Observation
---------------

In order to be able to define an action, at each time step :math:`t`, the environment provides the agent an observation :math:`O_{t}` which consists of a `NumPy <https://numpy.org/>`_ array with shapes :math:`(f, n, t)`, in which :math:`f` is the number of features in the agent state space, :math:`n` is the number of assets in the portfolio and :math:`t` is the size of the time series. The image below represents how the state space is generated based on all the time series.

.. image:: state_space.png
   :width: 800
   :alt: Diagram representing the generation of the state space.

In the code, :math:`f` is defined as the number of items in the :code:`features` argument, :math:`n` is the number of unique assets in :code:`tic_column` column of the input dataframe and :math:`t` can be changed in the :code:`time_window` argument.

The Reward
----------

The reward is given by the equation below:

.. math::

    r_{t} = ln \Bigl(\frac{V_{t}^{f}}{V_{t-1}^{f}}\Bigl),

in which :math:`V_{t}^{f}` is the value of the portfolio at the end of the current simulation step and :math:`V_{t}^{f}` is the value of the portfolio at the end of the last simulation step. In this formulation, every time a step reduces the value of the portfolio, a negative reward is provided and the opposite happens when a step increases the value of the portfolio.

Main methods
-------------

Just like any `Gymnasium <https://gymnasium.farama.org/index.html>`_ environment, POE has three main methods that can be used to interact with it.

**Reset method**
    The reset method is used to reset the environment to its initial state. It receives two arguments: a random seed and an options dictionary. Those arguments, however, are not used because the environment is deterministic and reset options are not implemented. Therefore, POE can be reset by simply doing:

    .. code-block::python

        obs, info = env.reset()

    :code:`obs` represents the initial observation of the environment (the observation that represents the initial state of the environment and that the agent will make use to take the first action) and :code:`info` is a dictionary with other information (such as the initial end final datetime of the observation, the data utilized, etc.)

**Step method**
    The step method is responsible for running a simulation step. It takes as input a `NumPy <https://numpy.org/>`_ of shape :code:`(n+1,)` representing the portfolio vector, in which :math:`n` is the size of the portfolio. Based on this input, the environment will calculate the effects of action performed in the market: the new portfolio value and distribution of the portfolio are calculated, for example. The step method can be used as follows:

    .. code-block::python
        action = np.array([0, 0.25, 0.15, 0.50, 0.1])

        obs, reward, terminal, truncated, info = env.step(action)

    :code:`obs` is the new observation generated after the simulation step, :code:`reward` is the numeric reward related to the step run, :code:`terminal` is true if the environment has reached a terminal state (the last datetime in the dataframe), :code:`truncated` is always false and only exists to respect the `Gymnasium <https://gymnasium.farama.org/index.html>`_ API and :code:`info` is the information dictionary. 

    .. note::

        If the environment is in a terminal state, the information dictionary will have a "metrics" key containing performance metrics calculated for the entire episode.

**render method**
    This method returns the current observation of the agent, so that it can be used to plot the training process.

