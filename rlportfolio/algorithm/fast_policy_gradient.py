from __future__ import annotations

import gymnasium as gym
from torch import nn
from torch.optim import AdamW, Optimizer
from tqdm import tqdm
from typing import Any

from rlportfolio.policy import EIIE
from rlportfolio.algorithm.policy_gradient import PolicyGradient
from rlportfolio.algorithm.buffers import ClearingReplayBuffer


class FastPolicyGradient(PolicyGradient):
    """Class implementing a faster version of the policy gradient algorithm to
    train portfolio optimization agents. This version can handle faster learning
    rates and perform fewer calculations.

    Note:
        During testing, the agent is optimized through online learning. The
        parameters of the policy is updated repeatedly after a constant period of
        time. To disable it, set the validation learning rate to 0.

    Attributes:
        train_env: Environment used to train the agent
        train_policy: Policy used in training.
        test_env: Environment used to test the agent.
        test_policy: Policy after test online learning.
    """

    def __init__(
        self,
        env: gym.Env,
        policy: type[nn.Module] = EIIE,
        policy_kwargs: dict[str, Any] = None,
        batch_size: int = 100,
        lr: float = 1e-2,
        polyak_avg_tau: float = 1,
        optimizer: type[Optimizer] = AdamW,
        use_tensorboard: bool = False,
        summary_writer_kwargs: dict[str, Any] = None,
        device: str = "cpu",
    ) -> FastPolicyGradient:
        """Initializes Fast Policy Gradient for portfolio optimization.

        Args:
            env: Training environment.
            policy: Policy architecture to be used.
            policy_kwargs: Arguments to be used in the policy network.
            batch_size: Batch size to train neural network.
            lr: policy neural network learning rate.
            optimizer: Optimizer of neural network.
            polyak_avg_tau: Tau parameter to be used in Polyak average (bigger than or equal 
                to 0 and smaller than or equal to 1). The bigger the parameter, the bigger 
                new training steps influence the target policy.
            use_tensorboard: If true, training logs will be added to tensorboard.
            summary_writer_kwargs: Arguments to be used in PyTorch's tensorboard summary
                writer.
            device: Device where neural network is run.
        """
        super().__init__(
            env=env,
            policy=policy,
            policy_kwargs=policy_kwargs,
            replay_buffer=ClearingReplayBuffer,
            batch_size=batch_size,
            lr=lr,
            polyak_avg_tau=polyak_avg_tau,
            optimizer=optimizer,
            use_tensorboard=use_tensorboard,
            summary_writer_kwargs=summary_writer_kwargs,
            device=device,
        )

    def train(
        self,
        episodes: int,
        val_period: int | None = None,
        val_env: gym.Env | None = None,
        val_batch_size: int | None = 10,
        val_lr: int | None = None,
        val_optimizer: type[Optimizer] | None = None,
        progress_bar: str | None = "permanent",
        name: str | None = None,
    ) -> tuple[dict[str, float] | None, dict[str, float] | None]:
        """Training sequence. The algorithm runs the specified number of episodes and
        after every batch_size simulation step, a gradient ascent is performed. After
        the gradient ascent, the replay buffer is cleared.

        Note:
            The validation step is run after every val_period episodes. This step simply
            runs an episode of the testing environment performing a gradient ascent after
            batch_size simulation steps, in order to apply online learning. To disable
            online learning, set learning rate to 0 or define a very big batch size.

        Args:
            episodes: Number of training episodes. (Training metrics are logged after
                every episode).
            val_period: Number of episodes to run before running a full episode in the
                validation environment and log metrics. If None, validation will happen
                in the end of all the training procedure.
            val_env: Validation environment. If None, no validation is performed.
            val_batch_size: Batch size to use in validation. If None, the training batch
                 size is used.
            val_lr: Learning rate to perform gradient ascent in validation. If None, the
                training learning rate is used instead.
            val_optimizer: Type of optimizer to use in the validation. If None, the same
                type used in training is set.
            progress_bar: If "permanent", a progress bar is displayed and is kept when
                completed. If "temporary", a progress bar is displayed but is deleted
                when completed. If None (or any other value), no progress bar is
                displayed.
            name: Name of the training sequence (it is displayed in the progress bar).

        Returns:
            The following tuple is returned: (metrics, val_metrics).

            metrics: Dictionary with metrics of the agent performance in the training
                environment. If None, no training was performed.
            val_metrics: Dictionary with metrics of the agent performance in the
                validation environment. If None, no validation was performed.
        """
        val_period = episodes if val_period is None else val_period

        # define tqdm arguments
        preffix, disable, leave = self._tqdm_arguments(progress_bar, name)

        # create metric variables
        metrics = None
        val_metrics = None

        # Start training
        for episode in (
            pbar := tqdm(
                range(1, episodes + 1),
                disable=disable,
                leave=leave,
                unit="episode",
            )
        ):
            # run and log episode
            pbar.colour = "white"
            pbar.set_description("{}Training agent".format(preffix))
            metrics = self._run_episode(
                gradient_steps=1,
                noise_index=episode,
                plot_loss_index=episode,
                update_rb=False,
            )
            self._plot_metrics(metrics, plot_index=episode, test=False)
            metrics.pop("rewards")
            pbar.set_postfix(self._tqdm_postfix_dict(metrics, val_metrics))

            # if there are remaining episodes in the buffer, update policy
            if self._can_update_policy(test=False, end_of_episode=True):
                self._gradient_ascent(noise_index=episode, update_rb=False)

            # validation step
            if val_env and episode % val_period == 0:
                pbar.colour = "yellow"
                pbar.set_description("{}Validating agent".format(preffix))
                val_metrics = self.test(
                    val_env,
                    policy=None,
                    batch_size=val_batch_size,
                    lr=val_lr,
                    optimizer=val_optimizer,
                    plot_index=int(episode / val_period),
                )

                pbar.set_postfix(self._tqdm_postfix_dict(metrics, val_metrics))

            if episode == episodes:
                pbar.colour = "green"
                pbar.set_description("{}Completed".format(preffix))

        return metrics, val_metrics

    def test(
        self,
        env: gym.Env,
        policy: nn.Module | None = None,
        batch_size: int | None = 10,
        lr: int | None = None,
        optimizer: type[Optimizer] | None = None,
        plot_index: int | None = None,
    ) -> dict[str, float]:
        """Tests the policy with online learning. The test sequence runs an episode of
        the environment and performs a gradient ascent after batch_size simulation steps
        in order to perform online learning. To disable online learning, set learning
        rate to 0 or set a very big batch size.

        Args:
            env: Environment to be used in testing.
            gradient_steps: Number of gradient ascent steps to perform after each
                simulation step.
            policy: Policy architecture to be used. If None, it will use the training
                architecture.
            batch_size: Batch size to train neural network. If None, it will use the
                training batch size.
            lr: Policy neural network learning rate. If None, it will use the training
                learning rate.
            optimizer: Optimizer of neural network. If None, it will use the training
                optimizer.
            plot_index: Index (x-axis) to be used to plot metrics. If None, no plotting
                is performed.

        Note:
            To disable online learning, set learning rate to 0 or a very big batch size.

        Returns:
            Dictionary with episode metrics.
        """

        return super().test(
            env,
            update_buffer=False,
            policy=policy,
            replay_buffer=ClearingReplayBuffer,
            batch_size=batch_size,
            lr=lr,
            optimizer=optimizer,
            plot_index=plot_index,
        )

    def _can_update_policy(
        self, test: bool = False, end_of_episode: bool = False
    ) -> bool:
        """Check if the conditions that allow a policy update are met.

        Args:
            test: If True, it uses the test parameters.
            end_of_episode: If True, it checks the conditions of the last update of
                an episode.

        Returns:
            True if policy update can happen.
        """
        buffer = self.test_buffer if test else self.train_buffer
        if end_of_episode and len(buffer) > 0:
            return True
        return super()._can_update_policy(test=test)
