from __future__ import annotations

from tqdm import tqdm

from rl_portfolio.algorithm.policy_gradient import PolicyGradient


class EpisodicPolicyGradient(PolicyGradient):
    """Class implementing policy gradient algorithm to train portfolio
    optimization agents.

    Note:
        During testing, the agent is optimized through online learning.
        The parameters of the policy is updated repeatedly after a constant
        period of time. To disable it, set learning rate to 0.

    Attributes:
        train_env: Environment used to train the agent
        train_policy: Policy used in training.
        test_env: Environment used to test the agent.
        test_policy: Policy after test online learning.
    """

    def train(
        self,
        episodes=100,
        gradient_steps=1,
        valid_period=None,
        valid_env=None,
        valid_gradient_steps=1,
        valid_use_train_buffer=True,
        valid_replay_buffer=None,
        valid_batch_size=None,
        valid_sample_bias=None,
        valid_sample_from_start=None,
        valid_lr=None,
        valid_optimizer=None,
    ):
        """Training sequence. The algorithm runs the specified number of episodes and
        after every simulation step, a defined number of gradient ascent steps are
        performed. This episodic version of policy gradient is suitable to
        non-deterministic environments whose observations or price-variations can differ
        in different episodes, since the replay buffer is completely updated when the
        algorithm rolls through the training data.

        Note:
            The validation step is run after every valid_period training steps. This
            step simply runs an episode of the testing environment performing
            valid_gradient_step training steps after each simulation step, in order
            to perform online learning. To disable online learning, set gradient steps
            or learning rate to 0, or set a very big batch size.

        Args:
            episodes: Number of training episodes. (Training metrics are logged after
                every episode).
            gradient_steps: Number of gradient ascent steps to perform after every
                simulation step of the episodes.
            valid_period: Number of episodes to run before running a full episode in the
                validation environment and log metrics. If None, validation will happen
                in the end of all the training procedure.
            valid_env: Validation environment. If None, no validation is performed.
            valid_gradient_steps: Number of gradient ascent steps to perform after
                each simulation step in the validation period.
            valid_use_train_buffer: If True, the validation period also makes use of
                experiences in the training replay buffer to perform online training.
                Set this option to True if the validation period is immediately after
                the training period.
            valid_replay_buffer: Type of replay buffer to use in validation. If None,
                it will be equal to the training replay buffer.
            valid_batch_size: Batch size to use in validation. If None, the training
                batch size is used.
            valid_sample_bias: Sample bias to be used if replay buffer is
                GeometricReplayBuffer. If None, the training sample bias is used.
            valid_sample_from_start: If True, the GeometricReplayBuffer will perform
                geometric distribution sampling from the beginning of the ordered
                experiences. If None, the training sample bias is used.
            valid_lr: Learning rate to perform gradient ascent in validation. If None,
                the training learning rate is used instead.
            valid_optimizer: Type of optimizer to use in the validation. If None, the
                same type used in training is set.
        """
        # If period is None, validations will only happen at the end of training.
        valid_period = episodes if valid_period is None else valid_period

        # Start training
        for episode in tqdm(range(1, episodes + 1)):
            # run and log episode
            metrics = self._run_episode(
                gradient_steps=gradient_steps, plot_loss_index=episode - 1
            )
            self._plot_metrics(metrics, plot_index=episode, test=False)

            # validation step
            if valid_env and episode % valid_period == 0:
                self.test(
                    valid_env,
                    gradient_steps=valid_gradient_steps,
                    use_train_buffer=valid_use_train_buffer,
                    policy=None,
                    replay_buffer=valid_replay_buffer,
                    batch_size=valid_batch_size,
                    sample_bias=valid_sample_bias,
                    sample_from_start=valid_sample_from_start,
                    lr=valid_lr,
                    optimizer=valid_optimizer,
                    plot_index=int(episode / valid_period),
                )
