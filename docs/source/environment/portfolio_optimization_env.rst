*PortfolioOptimizationEnv*
==========================

The *PortfolioOptimizationEnv* (or POE) is a `Gymnasium <https://gymnasium.farama.org/index.html>`_ environment implementing the mathematical formulation introduced in `A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem <https://doi.org/10.48550/arXiv.1706.10059>_` from Zhengyao Jiang and Dixing Xu and Jinjun Liang. 

Instantiating the environment
-----------------------------

Before using POE, it is necessary to provide the parameters it will use during the simulation. The most important one is the `Pandas <https://pandas.pydata.org/>`_ dataframe with historical data the environment will use to define the agent's observation.

For each asset in the portfolio, this dataframe must contain the 



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

In order to be able to define an action, at each time step :math:`t`, the environment provides the agent an observation :math:`O_{t}` which consists of a `NumPy <https://numpy.org/>`_ array with shapes :math:`(f, n, t)`, in which :math:`f` is the number of features in the agent state space, :math:`n` is the number of assets in the portfolio and :math:`t` is the size of the time series. 