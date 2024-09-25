What is an Environment?
=======================

The environment is a simulation which encompasses the dynamics of the market and calculatesd the effects of the agent's rebalancing in the portfolio value. It must be a `Gymnasium <https://gymnasium.farama.org/index.html>`_ object implementing the following methods:

* **reset:** Changes the environment's state to the initial state.
* **step:** Responsible for running a simulation step given an input action defined by the reinforcement learning agent.
* **render:** Returns informations that can be used to render the environment.

Currently, there is one environment implemented in the library called *PortfolioOptimizationEnv*, and it will be detailed in the next page.