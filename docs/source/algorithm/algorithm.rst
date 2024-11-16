What is an Algorithm?
=====================

The reinforcement learning algorithm is the core of the learning process: it is responsible to manage the interactions between the agent and the environment and to use its experiences to update the policy of actions in order to generate a relevant behavior.

In RLPortfolio, the API of the algorithms is composed of two main methods:

- The **train** method executes the training process.
- The **test** method applies a trained policy of actions in a different environment, in order to evaluate the agent's performance.

The user can create its own algorithms, but there are two state-of-the-art algorithms already implemented in RLPortfolio.