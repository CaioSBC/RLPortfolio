Episodic Policy Gradient
========================

The episodic policy gradient is a modified version of the :ref:`pg-label` in which the training sequence is merged with the execution of the environment's episode. In the original policy gradient, the agent executes a full episode and then, it not only interacts with it (except if performing a logging operation in the training environment). This is not a problem in deterministic environments such as :ref:`poe-label`, but it can be if the environment implements methods that simulate dynamic markets (such as state noise).

Therefore, to address this limitation, the episodic policy gradient algorithm was developed, changing the training sequence so that the the 6 steps of the training sequence of the :ref:`pg-label` are performed after the execution of every simulation step. Thus, the following training loop is implemented:

1. A simulation step is run, updating the replay buffer experiences.

2. The 6 training steps listed in :ref:`pg-label` are executed :code:`gradient_steps` times (this parameter can be set in the :code:`train` method).

This algorithm is very similar to the test sequence of :ref:`pg-label`. Additionally, The testing sequence, validation sequence is identical to the original Policy Gradient approach.

.. note::

    The algorithm is called episodic because it needs to run for :code:`episodes` number of episodes.

Logging
-------

A difference between this algorithm and :ref:`pg-label` is that there is no need to set a validation period, since several episodes are run and the logging is performed automatically.