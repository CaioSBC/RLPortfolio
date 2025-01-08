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

    def __init__(
        self,
        env: gym.Env,
        policy: type[nn.Module] = EIIE,
        policy_kwargs: dict[str, Any] = None,
        batch_size: int = 100,
        lr: float = 1e-3,
        optimizer: type[Optimizer] = AdamW,
        use_tensorboard: bool = False,
        summary_writer_kwargs: dict[str, Any] = None,
        device: str = "cpu",
    ):
        """Initializes Policy Gradient for portfolio optimization.

        Args:
          env: Training Environment.
          policy: Policy architecture to be used.
          policy_kwargs: Arguments to be used in the policy network.
          validation_env: Validation environment.
          batch_size: Batch size to train neural network.
          lr: policy Neural network learning rate.
          action_noise: Noise parameter (between 0 and 1) to be applied
            during training.
          optimizer: Optimizer of neural network.
          device: Device where neural network is run.
        """
        super().__init__(
            env=env,
            policy=policy,
            policy_kwargs=policy_kwargs,
            replay_buffer=ClearingReplayBuffer,
            batch_size=batch_size,
            lr=lr,
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
        """Training sequence.

        Args:
            episodes: Number of episodes to simulate.
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
        plot_index: int | None = None
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

        super().test(
            env,
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
