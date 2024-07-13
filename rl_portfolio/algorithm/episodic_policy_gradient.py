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
        val_period=None,
        val_env=None,
        val_gradient_steps=1,
        val_use_train_buffer=True,
        val_replay_buffer=None,
        val_batch_size=None,
        val_sample_bias=None,
        val_sample_from_start=None,
        val_lr=None,
        val_optimizer=None,
    ):
        """Training sequence. The algorithm runs the specified number of episodes and
        after every simulation step, a defined number of gradient ascent steps are
        performed. This episodic version of policy gradient is suitable to
        non-deterministic environments whose observations or price-variations can differ
        in different episodes, since the replay buffer is completely updated when the
        algorithm rolls through the training data.

        Note:
            The validation step is run after every val_period training steps. This
            step simply runs an episode of the testing environment performing
            val_gradient_step training steps after each simulation step, in order
            to perform online learning. To disable online learning, set gradient steps
            or learning rate to 0, or set a very big batch size.

        Args:
            episodes: Number of training episodes. (Training metrics are logged after
                every episode).
            gradient_steps: Number of gradient ascent steps to perform after every
                simulation step of the episodes.
            val_period: Number of episodes to run before running a full episode in the
                validation environment and log metrics. If None, validation will happen
                in the end of all the training procedure.
            val_env: Validation environment. If None, no validation is performed.
            val_gradient_steps: Number of gradient ascent steps to perform after each 
                simulation step in the validation period.
            val_use_train_buffer: If True, the validation period also makes use of
                experiences in the training replay buffer to perform online training.
                Set this option to True if the validation period is immediately after
                the training period.
            val_replay_buffer: Type of replay buffer to use in validation. If None, it 
                will be equal to the training replay buffer.
            val_batch_size: Batch size to use in validation. If None, the training batch
                 size is used.
            val_sample_bias: Sample bias to be used if replay buffer is
                GeometricReplayBuffer. If None, the training sample bias is used.
            val_sample_from_start: If True, the GeometricReplayBuffer will perform
                geometric distribution sampling from the beginning of the ordered
                experiences. If None, the training sample bias is used.
            val_lr: Learning rate to perform gradient ascent in validation. If None, the 
                training learning rate is used instead.
            val_optimizer: Type of optimizer to use in the validation. If None, the same 
                type used in training is set.
        """
        # If period is None, validations will only happen at the end of training.
        val_period = episodes if val_period is None else val_period

        # Start training
        for episode in tqdm(range(1, episodes + 1)):
            # run and log episode
            metrics = self._run_episode(
                gradient_steps=gradient_steps,
                noise_index=episode,
                plot_loss_index=episode,
            )
            self._plot_metrics(metrics, plot_index=episode, test=False)

            # validation step
            if val_env and episode % val_period == 0:
                self.test(
                    val_env,
                    gradient_steps=val_gradient_steps,
                    use_train_buffer=val_use_train_buffer,
                    policy=None,
                    replay_buffer=val_replay_buffer,
                    batch_size=val_batch_size,
                    sample_bias=val_sample_bias,
                    sample_from_start=val_sample_from_start,
                    lr=val_lr,
                    optimizer=val_optimizer,
                    plot_index=int(episode / val_period),
                )
