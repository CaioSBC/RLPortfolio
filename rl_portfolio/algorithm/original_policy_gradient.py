from __future__ import annotations

import copy

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import namedtuple

from rl_portfolio.architecture import EIIE
from rl_portfolio.algorithm.buffers import PortfolioVectorMemory
from rl_portfolio.algorithm.buffers import SequentialReplayBuffer
from rl_portfolio.algorithm.buffers import GeometricReplayBuffer
from rl_portfolio.utils import apply_action_noise
from rl_portfolio.utils import torch_to_numpy
from rl_portfolio.utils import numpy_to_torch
from rl_portfolio.utils import RLDataset


class PGPortfolio:
    """Class implementing the original PGPortfolio"""

    def __init__(
        self,
        env,
        policy=EIIE,
        policy_kwargs=None,
        replay_buffer=GeometricReplayBuffer,
        batch_size=100,
        sample_bias=1.0,
        sample_from_start=False,
        lr=1e-3,
        action_noise=0,
        parameter_noise=0,
        optimizer=AdamW,
        use_tensorboard=False,
        summary_writer_kwargs=None,
        device="cpu",
    ):
        """Initializes Policy Gradient for portfolio optimization.

        Args:
            env: Training Environment.
            policy: Policy architecture to be used.
            policy_kwargs: Arguments to be used in the policy network.
            validation_env: Validation environment.
            validation_kwargs: Arguments to be used in the validation step.
            replay_buffer: Class of replay buffer to be used to sample
                experiences in training.
            batch_size: Batch size to train neural network.
            sample_bias: Probability of success of a trial in a geometric
                distribution. Only used if buffer is GeometricReplayBuffer.
            sample_from_start: If True, will choose a sequence starting
                from the start of the buffer. Otherwise, it will start from
                the end. Only used if buffer is GeometricReplayBuffer.
            lr: policy Neural network learning rate.
            action_noise: Noise parameter (between 0 and 1) to be applied
                during training.
            parameter_noise: Standard deviation of gaussian noise applied
                policy network parameters.
            optimizer: Optimizer of neural network.
            use_tensorboard: If true, training logs will be added to
                tensorboard.
            summary_writer_kwargs: Arguments to be used in PyTorch's
                tensorboard summary writer.
            device: Device where neural network is run.
        """
        self.policy = policy
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.batch_size = batch_size
        self.sample_bias = sample_bias
        self.sample_from_start = sample_from_start
        self.lr = lr
        self.action_noise = action_noise
        self.parameter_noise = parameter_noise
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer

        self.summary_writer = None
        if use_tensorboard:
            summary_writer_kwargs = (
                {} if summary_writer_kwargs is None else summary_writer_kwargs
            )
            self.summary_writer = (
                SummaryWriter(**summary_writer_kwargs) if use_tensorboard else None
            )

        self.device = device
        if "device" in self.policy_kwargs:
            if self.policy_kwargs["device"] != self.device:
                if self.device == "cpu":
                    self.device = self.policy_kwargs["device"]
                else:
                    raise ValueError(
                        "Different devices set in algorithm ({}) and policy ({}) arguments".format(
                            self.device, self.policy_kwargs["device"]
                        )
                    )
        else:
            self.policy_kwargs["device"] = self.device

        self._setup_train(env)

    def _setup_train(self, env):
        """Initializes algorithm before training.

        Args:
          env: environment to be used in training.
        """
        # environment
        self.train_env = env

        # neural networks
        self.train_policy = self.policy(**self.policy_kwargs).to(self.device)
        self.train_optimizer = self.optimizer(
            self.train_policy.parameters(), lr=self.lr
        )

        # replay buffer and portfolio vector memory
        self.train_batch_size = self.batch_size
        self.train_buffer = self.replay_buffer(capacity=env.episode_length)
        self.train_pvm = PortfolioVectorMemory(env.episode_length, env.portfolio_size)

        # dataset and dataloader
        dataset = RLDataset(
            self.train_buffer, self.batch_size, self.sample_bias, self.sample_from_start
        )
        self.train_dataloader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )

    def _run_episode(self, test=False, gradient_ascent=False):
        """Runs a full episode (the agent rolls through all the environment's
        data).

        Args:
            test: If True, the episode is running during a test routine.
            gradient_ascent: If True, the agent will perform a gradient ascent
                after each simulation step (online learning).

        """
        if test:
            obs, info = self.test_env.reset()  # observation
            self.test_pvm.reset()  # reset portfolio vector memory
        else:
            obs, info = self.train_env.reset()  # observation
            self.train_pvm.reset()  # reset portfolio vector memory
        done = False
        metrics = {"rewards": []}
        index = 0
        while not done:
            # define policy input tensors
            last_action = (
                self.test_pvm.retrieve() if test else self.train_pvm.retrieve()
            )
            obs_batch = numpy_to_torch(obs, add_batch_dim=True, device=self.device)
            last_action_batch = numpy_to_torch(
                last_action, add_batch_dim=True, device=self.device
            )

            # define action
            action = torch_to_numpy(
                self.train_policy(obs_batch, last_action_batch), squeeze=True
            )

            # update portfolio vector memory
            self.test_pvm.add(action) if test else self.train_pvm.add(action)

            # run simulation step
            next_obs, reward, done, _, info = (
                self.test_env.step(action) if test else self.train_env.step(action)
            )

            # add experience to replay buffer
            exp = (obs, last_action, info["price_variation"], index)
            self.test_buffer.add(exp) if test else self.train_buffer.add(exp)
            index += 1

            # log rewards
            metrics["rewards"].append(reward)

            # if episode ended, get metrics to log
            if "metrics" in info:
                metrics.update(info["metrics"])

            # update policy networks
            if gradient_ascent and self._can_update_policy(test=test):
                self._gradient_ascent(test=test, update_buffers=False)

            obs = next_obs

        return metrics

    def train(
        self,
        steps=10000,
        logging_period=250,
        validation_period=None,
        validation_env=None,
        validation_replay_buffer=None,
        validation_batch_size=None,
        validation_sample_bias=None,
        validation_sample_from_start=None,
        validation_lr=None,
        validation_optimizer=None,
    ):
        """Training sequence.

        Args:
            steps: Number of training steps.
            logging_period: Number of training steps to perform gradient ascent
                before running a full episode and log the agent's metrics.
            validation_period: Number of training steps to perform before running
                a full episode in the validation environment and log metrics. If
                None, no validation is done.
            validation_env: Validation environment. If None, no validation is
                performed.
            validation_replay_buffer: Type of replay buffer to use in validation.
                If None, it will be equal to the training replay buffer.
            validation_batch_size: Batch size to use in validation. If None, the
                training batch size is used.
            validation_sample_bias: Sample bias to be used if replay buffer is
                GeometricReplayBuffer. If None, the training sample bias is used.
            validation_sample_from_start: If True, the GeometricReplayBuffer will
                perform geometric distribution sampling from the beginning of the
                ordered experiences. If None, the training sample bias is used.
            validation_lr: Learning rate to perform gradient ascent in validation.
                If None, the training learning rate is used instead.
            validation_optimizer: Type of optimizer to use in the validation. If
                None, the same type used in training is set.
        """
        # If periods are None, loggings and validations will only happen at
        # the end of training.
        logging_period = steps if logging_period is None else logging_period
        validation_period = steps if validation_period is None else validation_period

        # run the episode to fill the buffers
        self._run_episode()

        # Start training
        for step in tqdm(range(1, steps + 1)):
            if self._can_update_policy():
                policy_loss = self._gradient_ascent()

                # plot policy loss in tensorboard
                self._plot_loss(policy_loss, step)

                # run episode to log metrics
                if step % logging_period == 0:
                    metrics = self._run_episode()
                    self._plot_metrics(
                        metrics, plot_index=int(step / logging_period), test=False
                    )

                # validation step
                if validation_env and step % validation_period == 0:
                    self.test(
                        validation_env,
                        validation_replay_buffer,
                        validation_batch_size,
                        validation_sample_bias,
                        validation_sample_from_start,
                        validation_lr,
                        validation_optimizer,
                        plot_index=int(step / validation_period),
                    )

    def _setup_test(
        self,
        env,
        policy,
        replay_buffer,
        batch_size,
        sample_bias,
        sample_from_start,
        lr,
        optimizer,
    ):
        """Initializes algorithm before testing.

        Args:
            env: Environment.
            policy: Policy architecture to be used.
            replay_buffer: Class of replay buffer to be used.
            batch_size: Batch size to train neural network.
            sample_bias: Probability of success of a trial in a geometric distribution.
                Only used if buffer is GeometricReplayBuffer.
            sample_from_start: If True, will choose a sequence starting from the start
                of the buffer. Otherwise, it will start from the end. Only used if
                buffer is GeometricReplayBuffer.
            lr: Policy neural network learning rate.
            optimizer: Optimizer of neural network.
        """
        # environment
        self.test_env = env

        # process None arguments
        policy = self.train_policy if policy is None else policy
        replay_buffer = self.replay_buffer if replay_buffer is None else replay_buffer
        batch_size = self.batch_size if batch_size is None else batch_size
        sample_bias = self.sample_bias if sample_bias is None else sample_bias
        sample_from_start = (
            self.sample_from_start if sample_from_start is None else sample_from_start
        )
        lr = self.lr if lr is None else lr
        optimizer = self.optimizer if optimizer is None else optimizer

        # define policy
        self.test_policy = copy.deepcopy(policy).to(self.device)
        self.test_optimizer = optimizer(self.test_policy.parameters(), lr=lr)

        # replay buffer and portfolio vector memory
        self.test_batch_size = batch_size
        self.test_buffer = replay_buffer(capacity=env.episode_length)
        self.test_pvm = PortfolioVectorMemory(env.episode_length, env.portfolio_size)

        # dataset and dataloader
        dataset = RLDataset(
            self.test_buffer, self.test_batch_size, sample_bias, sample_from_start
        )
        self.test_dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

    def test(
        self,
        env,
        policy=None,
        replay_buffer=None,
        batch_size=None,
        sample_bias=None,
        sample_from_start=None,
        lr=None,
        optimizer=None,
        plot_index=None,
    ):
        """Tests the policy with online learning.

        Args:
            env: Environment to be used in testing.
            policy: Policy architecture to be used. If None, it will use the training
                architecture.
            replay_buffer: Class of replay buffer to be used. If None, it will use the
                training replay buffer.
            batch_size: Batch size to train neural network. If None, it will use the
                training batch size.
            sample_bias: Probability of success of a trial in a geometric distribution.
                Only used if buffer is GeometricReplayBuffer. If None, it will use the
                training sample bias.
            sample_from_start: If True, will choose a sequence starting from the start
                of the buffer. Otherwise, it will start from the end. Only used if
                buffer is GeometricReplayBuffer. If None, it will use the training
                sample_from_start.
            lr: Policy neural network learning rate. If None, it will use the training
                learning rate.
            optimizer: Optimizer of neural network. If None, it will use the training
                optimizer.
            plot_index: Index (x-axis) to be used to plot metrics. If None, no plotting
                is performed.

        Note:
            To disable online learning, set learning rate to 0 or a very big batch size.
        """
        self._setup_test(
            env,
            policy,
            replay_buffer,
            batch_size,
            sample_bias,
            sample_from_start,
            lr,
            optimizer,
        )

        # run episode performing gradient ascent after each simulation step (online learning)
        metrics = self._run_episode(test=True, gradient_ascent=True)

        # log test metrics
        if plot_index is not None:
            self._plot_metrics(metrics, plot_index, test=True)

    def _gradient_ascent(self, test=False, update_buffers=True):
        """Performs the gradient ascent step in the policy gradient algorithm.

        Args:
            test: If true, it uses the test dataloader and policy.
            update_buffers: If True, portfolio vector memory and replay buffers
                will be updated.

        Returns:
            Negative of policy loss (since it's gradient ascent).
        """
        # get batch data from dataloader
        obs, last_actions, price_variations, indexes = (
            next(iter(self.test_dataloader))
            if test
            else next(iter(self.train_dataloader))
        )
        obs = obs.to(self.device)
        last_actions = last_actions.to(self.device)
        price_variations = price_variations.to(self.device)

        # define agent's actions and apply noise.
        actions = (
            self.test_policy(obs, last_actions)
            if test
            else apply_action_noise(
                self.train_policy(obs, last_actions), epsilon=self.action_noise
            )
        )

        # calculate comission rate and transaction remainder factor
        comission_rate = (
            self.test_env._comission_fee_pct
            if test
            else self.train_env._comission_fee_pct
        )
        with torch.no_grad():
            trf_mu = 1 - comission_rate * torch.sum(
                torch.abs(actions[:, 1:] - last_actions[:, 1:]), dim=1, keepdim=True
            )

        # define policy loss (negative for gradient ascent)
        policy_loss = -torch.mean(
            torch.log(torch.sum(actions * price_variations * trf_mu, dim=1))
        )

        # update policy network
        if test:
            self.test_policy.zero_grad()
            policy_loss.backward()
            self.test_optimizer.step()
        else:
            self.train_policy.zero_grad()
            policy_loss.backward()
            self.train_optimizer.step()

        # actions can be updated in the buffers and memories
        if update_buffers:
            self._update_buffers(actions, indexes, test)

        return -policy_loss

    def _can_update_policy(self, test=False, end_of_episode=False):
        """Check if the conditions that allow a policy update are met.

        Args:
            test: If True, it uses the test parameters.
            end_of_episode: If True, it checks the conditions of the last
                update of an episode.

        Returns:
            True if policy update can happen.
        """
        buffer = self.test_buffer if test else self.train_buffer
        batch_size = self.test_batch_size if test else self.train_batch_size
        if (
            isinstance(buffer, SequentialReplayBuffer)
            and end_of_episode
            and len(buffer) > 0
        ):
            return True
        if len(buffer) >= batch_size and not end_of_episode:
            return True
        return False

    def _update_buffers(self, actions, indexes, test):
        """Updates the portfolio vector memory and the replay buffers
        considering the actions taken during gradient ascent.

        Args:
            actions: Batch of performed actions with shape (batch_size,
                action_size).
            indexes: Batch with the indices of the batch data used in
                in the gradient ascent. Shape is (batch_size,).
            test: If True, test buffers must be updated.
        """
        actions = list(torch_to_numpy(actions))
        buffer_indexes = (indexes + 1).tolist()
        pvm_indexes = indexes.tolist()

        if test:
            # update portfolio vector memory
            self.test_pvm.add_at(actions, pvm_indexes)
            if buffer_indexes[-1] >= len(self.test_buffer):
                actions.pop()
                buffer_indexes.pop()
            # update replay buffer last action value
            self.test_buffer.update_value(actions, buffer_indexes, 1)
        else:
            # update portfolio vector memory
            self.train_pvm.add_at(actions, pvm_indexes)
            if buffer_indexes[-1] >= len(self.train_buffer):
                actions.pop()
                buffer_indexes.pop()
            # update replay buffer last action value
            self.train_buffer.update_value(actions, buffer_indexes, 1)

    def _plot_loss(self, loss, plot_index):
        """Plots the policy loss in tensorboard.

        Args:
            loss: The value of the policy loss.
            plot_index: Index (x-axis) to be used to plot the loss
        """
        if self.summary_writer:
            self.summary_writer.add_scalar("Loss/Train", loss, plot_index)

    def _plot_metrics(self, metrics, plot_index, test):
        """Plots the metrics calculated after an episode in tensorboard.

        Args:
            metrics: Dictionary containing the calculated metrics.
            plot_index: Index (x-axis) to be used to plot metrics.
            test: If True, metrics from a testing episode are being used.
        """
        context = "Test" if test else "Train"
        if self.summary_writer:
            self.summary_writer.add_scalar(
                "Final Accumulative Portfolio Value/{}".format(context),
                metrics["fapv"],
                plot_index,
            )
            self.summary_writer.add_scalar(
                "Maximum DrawDown/{}".format(context), metrics["mdd"], plot_index
            )
            self.summary_writer.add_scalar(
                "Sharpe Ratio/{}".format(context), metrics["sharpe"], plot_index
            )
            self.summary_writer.add_scalar(
                "Mean Reward/{}".format(context),
                np.mean(metrics["rewards"]),
                plot_index,
            )
