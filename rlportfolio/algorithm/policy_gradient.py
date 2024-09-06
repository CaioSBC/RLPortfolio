from __future__ import annotations

import copy
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Any, Callable

from rlportfolio.policy import EIIE
from rlportfolio.algorithm.buffers import PortfolioVectorMemory
from rlportfolio.algorithm.buffers import SequentialReplayBuffer
from rlportfolio.algorithm.buffers import GeometricReplayBuffer
from rlportfolio.utils import apply_action_noise
from rlportfolio.utils import combine_replay_buffers
from rlportfolio.utils import torch_to_numpy
from rlportfolio.utils import numpy_to_torch
from rlportfolio.utils import RLDataset


class PolicyGradient:
    """Class implementing policy gradient algorithm to train portfolio
    optimization agents. This class implements the work introduced in the
    following article: https://doi.org/10.48550/arXiv.1706.10059.

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
        replay_buffer: type[SequentialReplayBuffer] = GeometricReplayBuffer,
        batch_size: int = 100,
        sample_bias: float = 1.0,
        sample_from_start: bool = False,
        lr: float = 1e-3,
        action_noise: str | None = None,
        action_epsilon: float | Callable[[int], float] = 0,
        action_alpha: float | Callable[[int], float] = 1.0,
        parameter_noise: float | Callable[[int], float] = 0,
        optimizer: type[Optimizer] = AdamW,
        use_tensorboard: bool = False,
        summary_writer_kwargs: dict[str, Any] = None,
        device: str = "cpu",
    ) -> PolicyGradient:
        """Initializes Policy Gradient for portfolio optimization.

        Args:
            env: Training environment.
            policy: Policy architecture to be used.
            policy_kwargs: Arguments to be used in the policy network.
            validation_env: Validation environment.
            validation_kwargs: Arguments to be used in the validation step.
            replay_buffer: Class of replay buffer to be used to sample experiences in
                training.
            batch_size: Batch size to train neural network.
            sample_bias: Probability of success of a trial in a geometric distribution.
                Only used if buffer is GeometricReplayBuffer.
            sample_from_start: If True, will choose a sequence starting from the start
                of the buffer. Otherwise, it will start from the end. Only used if
                buffer is GeometricReplayBuffer.
            lr: policy neural network learning rate.
            action_noise: Name of the model to be used in the action noise. The options are
                "logarithmic", "logarithmic_const", "dirichlet" or None. If None, no action 
                noise is applied.
            action_epsilon: Noise logarithmic parameter (bigger than or equal to 0) to be
                applied to performed actions during training. It can be a value or a
                function whose argument is the number of training episodes/steps and that
                outputs the noise value.
            action_alpha: Alpha parameter (bigger than 1) to be used to create a Dirichlet 
                distribution in the "dirichlet" noise model. It can be a value or a function 
                whose argument is the number of training episodes/steps and that outputs the
                noise value.
            parameter_noise: Noise parameter (bigger than or equal to 0) to be applied
                to the parameters of the policy network during training. It can be a
                value or a function whose argument is the number of training episodes/
                steps and that outputs the noise value. Currently not implemented.
            optimizer: Optimizer of neural network.
            use_tensorboard: If true, training logs will be added to tensorboard.
            summary_writer_kwargs: Arguments to be used in PyTorch's tensorboard summary
                writer.
            device: Device where neural network is run.
        """
        self.policy = policy
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.batch_size = batch_size
        self.sample_bias = sample_bias
        self.sample_from_start = sample_from_start
        self.lr = lr
        self.action_noise = action_noise
        self.action_epsilon = action_epsilon
        self.action_alpha = action_alpha
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

    def _setup_train(self, env: gym.Env) -> None:
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

    def _run_episode(
        self,
        test: bool = False,
        gradient_steps: int = 0,
        initial_index: int = 0,
        noise_index: int | None = None,
        plot_loss_index: int | None = None,
    ) -> dict[str, float | list[float]]:
        """Runs a full episode (the agent rolls through all the environment's data).
        At the end of each simuloation step, the agent can perform a number of gradient
        ascent operations if specified in the arguments.

        Args:
            test: If True, the episode is running during a test routine.
            gradient_steps: The number of gradient ascent operations the agent will
                perform after each simulation step (online learning).
            initial_index: Initial index value of the simulation step. It is used when
                the replay buffer is pre-filled with experiences and its capacity is
                bigger than the episode length.
            noise_index: Index value to be used in case of a callable as noise value. If
                None, an exception might be raised if action noise is callable.
            plot_loss_index: Index value to be used to log the policy loss. If None, no
                logging is performed.

        Returns:
            Dictionary with episode metrics.
        """
        if test:
            obs, info = self.test_env.reset()  # observation
            self.test_pvm.reset()  # reset portfolio vector memory
            plot_loss_index = (  # initial value of log loss index
                (plot_loss_index - 1) * self.test_env.episode_length + 1
                if plot_loss_index is not None
                else plot_loss_index
            )
        else:
            obs, info = self.train_env.reset()  # observation
            self.train_pvm.reset()  # reset portfolio vector memory
            plot_loss_index = (
                (plot_loss_index - 1) * self.train_env.episode_length + 1
                if plot_loss_index is not None
                else plot_loss_index
            )
        done = False
        metrics = {"rewards": []}
        index = initial_index
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
            if gradient_steps > 0 and self._can_update_policy(test=test):
                for i in range(gradient_steps):
                    policy_loss = self._gradient_ascent(
                        test=test, noise_index=noise_index
                    )
                    if plot_loss_index is not None:
                        self._plot_loss(policy_loss, plot_loss_index)
                        plot_loss_index += 1

            obs = next_obs

        return metrics

    def _tqdm_arguments(
        self, progress_bar: str | None, name: str | None
    ) -> tuple[str, bool, bool]:
        """Parses tqdm arguments to training progress bar.

        Args:
            progress_bar: If "permanent", a progress bar is displayed and is kept when
                completed. If "temporary", a progress bar is displayed but is deleted
                when completed. If None (or any other value), no progress bar is
                displayed.
            name: Name of the training sequence (it is displayed in the progress bar).

        Returns:
            The following tuple is returned: (preffix, disable, leave).

            preffix: Preffix to be added to tqdm desc argument.
            disable: Value of tqdm disable argument.
            leave: Value of tqdm leave argument
        """
        if name is None:
            preffix = ""
        else:
            preffix = "" if name == "" else "{} - ".format(name)

        if progress_bar == "permanent":
            disable = False
            leave = True
        elif progress_bar == "temporary":
            disable = False
            leave = False
        else:
            disable = True
            leave = False

        return preffix, disable, leave

    def _tqdm_postfix_dict(
        self, metrics: dict[str, Any], val_metrics: dict[str, Any]
    ) -> dict[str, float]:
        """Create tqdm postfix dictionary to print in progress bar.

        Args:
            metrics: Dictionary with metrics of training period.
            val_metrics: Dictionary with metrics of validation period.

        Returns:
            Dictionary with metrics to print progress bar postfix.
        """
        dict_ = {}
        if metrics is not None:
            dict_.update(metrics)
            dict_.pop("value")
        if val_metrics is not None:
            new_val_metrics = {
                "val_" + str(key): val for key, val in val_metrics.items()
            }
            dict_.update(new_val_metrics)
            dict_.pop("val_value")
        return dict_

    def train(
        self,
        steps: int,
        logging_period: int | None = None,
        val_period: int | None = None,
        val_env: gym.Env | None = None,
        val_gradient_steps: int = 1,
        val_use_train_buffer: bool = True,
        val_replay_buffer: type[SequentialReplayBuffer] | None = None,
        val_batch_size: int | None = None,
        val_sample_bias: float | None = None,
        val_sample_from_start: bool | None = None,
        val_lr: float | None = None,
        val_optimizer: type[Optimizer] | None = None,
        progress_bar: str | None = "permanent",
        name: str | None = None,
    ) -> tuple[dict[str, float] | None, dict[str, float] | None]:
        """Training sequence. Initially, the algorithm runs a full episode without
        any training in order to full replay buffers. Then, several training steps
        are executed using data from the replay buffer in order to maximize the
        objective function. At the end of each training step, the buffer is updated
        with new outputs of the policy network.

        Note:
            The validation step is run after every val_period training steps. This
            step simply runs an episode of the testing environment performing
            val_gradient_step training steps after each simulation step, in order
            to perform online learning. To disable online learning, set gradient steps
            or learning rate to 0, or set a very big batch size.

        Args:
            steps: Number of training steps.
            logging_period: Number of training steps to perform gradient ascent
                before running a full episode and log the agent's metrics. If None,
                logging will be performed in the end of all the training procedure.
            val_period: Number of training steps to perform before running a full
                episode in the validation environment and log metrics. If None,
                validation will happen in the end of all the training procedure.
            val_env: Validation environment. If None, no validation is performed.
            val_gradient_steps: Number of gradient ascent steps to perform after
                each simulation step in the validation period.
            val_use_train_buffer: If True, the validation period also makes use of
                experiences in the training replay buffer to perform online training.
                Set this option to True if the validation period is immediately after
                the training period.
            val_replay_buffer: Type of replay buffer to use in validation. If None,
                it will be equal to the training replay buffer.
            val_batch_size: Batch size to use in validation. If None, the training
                batch size is used.
            val_sample_bias: Sample bias to be used if replay buffer is
                GeometricReplayBuffer. If None, the training sample bias is used.
            val_sample_from_start: If True, the GeometricReplayBuffer will perform
                geometric distribution sampling from the beginning of the ordered
                experiences. If None, the training sample bias is used.
            val_lr: Learning rate to perform gradient ascent in validation. If None,
                the training learning rate is used instead.
            val_optimizer: Type of optimizer to use in the validation. If None, the
                same type used in training is set.
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
        # If periods are None, loggings and validations will only happen at
        # the end of training.
        logging_period = steps if logging_period is None else logging_period
        val_period = steps if val_period is None else val_period

        # define tqdm arguments
        preffix, disable, leave = self._tqdm_arguments(progress_bar, name)

        # create metric variables
        metrics = None
        val_metrics = None

        # Start training
        for step in (
            pbar := tqdm(
                range(1, steps + 1),
                disable=disable,
                leave=leave,
                unit="step",
            )
        ):
            # in the first step, fill the buffers.
            if step == 1:
                pbar.set_description("{}Filling replay buffer".format(preffix))
                self._run_episode()

            pbar.colour = "white"
            pbar.set_description("{}Training agent".format(preffix))
            if self._can_update_policy():
                policy_loss = self._gradient_ascent(noise_index=step)

                # plot policy loss in tensorboard
                self._plot_loss(policy_loss, step)

                # run episode to log metrics
                if step % logging_period == 0:
                    pbar.colour = "blue"
                    pbar.set_description("{}Logging metrics".format(preffix))
                    metrics = self._run_episode()
                    self._plot_metrics(
                        metrics, plot_index=int(step / logging_period), test=False
                    )
                    metrics.pop("rewards")

                    pbar.set_postfix(self._tqdm_postfix_dict(metrics, val_metrics))

                # validation step
                if val_env and step % val_period == 0:
                    pbar.colour = "yellow"
                    pbar.set_description("{}Validating agent".format(preffix))
                    val_metrics = self.test(
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
                        plot_index=int(step / val_period),
                    )
                    if "rewards" in val_metrics:
                        val_metrics.pop("rewards")

                    pbar.set_postfix(self._tqdm_postfix_dict(metrics, val_metrics))

            if step == steps:
                pbar.colour = "green"
                pbar.set_description("{}Completed".format(preffix))

        return metrics, val_metrics

    def _setup_test(
        self,
        env: gym.Env,
        use_train_buffer: bool,
        policy: nn.Module,
        replay_buffer: type[SequentialReplayBuffer],
        batch_size: int,
        sample_bias: float,
        sample_from_start: bool,
        lr: float,
        optimizer: type[Optimizer],
    ) -> None:
        """Initializes algorithm before testing.

        Args:
            env: Environment.
            use_train_buffer: If True, the test period also makes use of experiences in
                the training replay buffer to perform online training. Set this option
                to True if the test period is immediately after the training period.
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
        if use_train_buffer:
            self.test_buffer = combine_replay_buffers(
                [self.train_buffer, self.test_buffer], replay_buffer
            )
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
        env: gym.Env,
        gradient_steps: int = 1,
        use_train_buffer: bool = False,
        policy: nn.Module | None = None,
        replay_buffer: SequentialReplayBuffer | None = None,
        batch_size: int | None = None,
        sample_bias: float | None = None,
        sample_from_start: bool | None = None,
        lr: float = None,
        optimizer: type[Optimizer] | None = None,
        plot_index: int | None = None,
    ) -> dict[str, float]:
        """Tests the policy with online learning. The test sequence runs an episode of
        the environment and performs gradient_step training steps after each simulation
        step in order to perform online learning. To disable online learning, set gradient
        steps or learning rate to 0, or set a very big batch size.

        Args:
            env: Environment to be used in testing.
            gradient_steps: Number of gradient ascent steps to perform after each
                simulation step.
            use_train_buffer: If True, the test period also makes use of experiences in
                the training replay buffer to perform online training. Set this option
                to True if the test period is immediately after the training period.
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

        Returns:
            Dictionary with episode metrics.
        """
        self._setup_test(
            env,
            use_train_buffer,
            policy,
            replay_buffer,
            batch_size,
            sample_bias,
            sample_from_start,
            lr,
            optimizer,
        )

        # run episode performing gradient ascent after each simulation step (online learning)
        initial_index = self.train_buffer.capacity if use_train_buffer else 0
        metrics = self._run_episode(
            test=True, gradient_steps=gradient_steps, initial_index=initial_index
        )

        # log test metrics
        if plot_index is not None:
            self._plot_metrics(metrics, plot_index, test=True)

        # remove rewards list from metrics
        if "rewards" in metrics:
            metrics.pop("rewards")
        return metrics

    def _gradient_ascent(
        self,
        test: bool = False,
        noise_index: int | None = None,
        update_rb: bool = True,
        update_pvm: bool = False,
    ) -> float:
        """Performs the gradient ascent step in the policy gradient algorithm.

        Args:
            test: If true, it uses the test dataloader and policy.
            noise_index: Index value to be used in case of a callable as noise value.
            update_rb: If True, replay buffers will be updated after gradient ascent.
            update_pvm: If True, portfolio vector memories will be updated after
                gradient ascent.

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

        # define agent's actions
        if test:
            actions = self.test_policy(obs, last_actions)
        else:
            # define action noise.
            if callable(self.action_epsilon):
                if noise_index is None:
                    raise TypeError(
                        "Noise index parameter of callable action noise epsilon is None."
                    )
                action_epsilon = self.action_epsilon(noise_index)
            else:
                action_epsilon = self.action_epsilon
            if callable(self.action_alpha):
                if noise_index is None:
                    raise TypeError(
                        "Noise index parameter of callable action noise gamma is None."
                    )
                action_alpha = self.action_alpha(noise_index)
            else:
                action_alpha = self.action_alpha
            actions = apply_action_noise(
                self.train_policy(obs, last_actions),
                noise_model=self.action_noise,
                epsilon=action_epsilon,
                alpha=action_alpha,
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
        self._update_buffers(actions, indexes, test, update_rb, update_pvm)

        return -policy_loss

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

    def _update_buffers(
        self,
        actions: torch.Tensor,
        indexes: torch.Tensor,
        test: bool,
        update_rb: bool = True,
        update_pvm: bool = False,
    ) -> None:
        """Updates the portfolio vector memory and the replay buffers considering the
        actions taken during gradient ascent.

        Args:
            actions: Batch of performed actions with shape (batch_size, action_size).
            indexes: Batch with the indices of the batch data used in in the gradient
                ascent. Shape is (batch_size,).
            test: If True, test buffers must be updated.
            update_rb: If True, updates replay buffers.
            update_pvm: If True, updates portfolio vector memories.
        """
        if not update_rb and not update_pvm:
            return
        actions = list(torch_to_numpy(actions))
        indexes = (indexes + 1).tolist()

        if test:
            if update_pvm:
                # update portfolio vector memory
                self.test_pvm.add_at(actions, indexes)
            if update_rb:
                if indexes[-1] >= len(self.test_buffer):
                    actions.pop()
                    indexes.pop()
                # update replay buffer last action value
                self.test_buffer.update_value(actions, indexes, 1)
        else:
            # update portfolio vector memory
            if update_pvm:
                # update portfolio vector memory
                self.train_pvm.add_at(actions, indexes)
            if update_rb:
                if indexes[-1] >= len(self.train_buffer):
                    actions.pop()
                    indexes.pop()
                # update replay buffer last action value
                self.train_buffer.update_value(actions, indexes, 1)

    def _plot_loss(self, loss: float, plot_index: int) -> None:
        """Plots the policy loss in tensorboard.

        Args:
            loss: The value of the policy loss.
            plot_index: Index (x-axis) to be used to plot the loss
        """
        if self.summary_writer:
            self.summary_writer.add_scalar("Loss/Train", loss, plot_index)

    def _plot_metrics(
        self, metrics: dict[str, float], plot_index: int, test: bool
    ) -> None:
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
