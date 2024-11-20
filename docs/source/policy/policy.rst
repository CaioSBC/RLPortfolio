What is a Policy?
=================

In reinforcement learning, a policy of action is the set of rules that defines the agent's behavior given its perceived state. Therefore, at time step :math:`t`, the policy must be able to process the current state :math:`S_{t}` and output the action :math:`A_{t}` to be performed. It can be a set of rules, a mathematical function and, in the context of deep reinforcement learning, a deep neural network.

Policies can be:

- **deterministic**: In this case, for each possible state, the policy chooses the same action.
- **stochastic**: The policy can determine different actions for the same input state. Usually, this approach generates a probability distribution for all possible actions. 


In the context of portfolio optimization, deterministic policies are used and they process not only the current state of the agent :math:`S_{t}`, but also its last performed action :math:`A_{t-1}` so that it is possible to infer losses related to brokerage fees.

In RLPortfolio, a policy can be any `PyTorch <https://pytorch.org/>`_ module which, in its forward propagation, process the current state and the last performed action. Thus, the action performed by the agent is :math:`A_{t} = \theta(S_{t}, A_{t-1})`, in which :math:`\theta` represents the neural network used as policy.

The user can create its own custom policy or utilize the ones already implemented in this library.