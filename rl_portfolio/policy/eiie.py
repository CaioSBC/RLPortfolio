from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from torch.func import stack_module_state, functional_call


class EIIE(nn.Module):
    def __init__(
        self,
        initial_features: int = 3,
        k_size: int = 3,
        conv_mid_features: int = 2,
        conv_final_features: int = 20,
        time_window: int = 50,
        device: str = "cpu",
    ) -> EIIE:
        """Convolutional EIIE (ensemble of identical independent evaluators) policy
        network initializer.

        Args:
            initial_features: Number of input features.
            k_size: Size of first convolutional kernel.
            conv_mid_features: Size of intermediate convolutional channels.
            conv_final_features: Size of final convolutional channels.
            time_window: Size of time window used as agent's state.
            device: Device in which the neural network will be run.

        Note:
            Reference article: https://doi.org/10.48550/arXiv.1706.10059.
        """
        super().__init__()
        self.device = device

        n_size = time_window - k_size + 1

        self.sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features,
                out_channels=conv_mid_features,
                kernel_size=(1, k_size),
                device=self.device,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_size),
                device=self.device,
            ),
            nn.ReLU(),
        )

        self.final_convolution = nn.Conv2d(
            in_channels=conv_final_features + 1,
            out_channels=1,
            kernel_size=(1, 1),
            device=self.device,
        )

        self.softmax = nn.Sequential(nn.Softmax(dim=-1))

    def forward(
        self, observation: torch.Tensor, last_action: torch.Tensor
    ) -> torch.Tensor:
        """Policy network's forward propagation. Defines a most favorable
        action of this policy given the inputs.

        Args:
            observation: environment observation.
            last_action: Last action performed by agent.

        Returns:
            Action to be taken.
        """
        last_stocks, cash_bias = self._process_last_action(last_action)
        cash_bias = torch.zeros_like(cash_bias).to(self.device)

        output = self.sequential(observation)  # shape [N, 20, PORTFOLIO_SIZE, 1]
        output = torch.cat(
            [last_stocks, output], dim=1
        )  # shape [N, 21, PORTFOLIO_SIZE, 1]
        output = self.final_convolution(output)  # shape [N, 1, PORTFOLIO_SIZE, 1]
        output = torch.cat(
            [cash_bias, output], dim=2
        )  # shape [N, 1, PORTFOLIO_SIZE + 1, 1]

        # output shape must be [N, features] = [1, PORTFOLIO_SIZE + 1], being N batch size (1)
        # and size the number of features (weights vector).
        output = torch.squeeze(output, 3)
        output = torch.squeeze(output, 1)  # shape [N, PORTFOLIO_SIZE + 1]

        output = self.softmax(output)

        return output

    def _process_last_action(
        self, last_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process the last action to retrieve cash bias and last stocks.

        Args:
          last_action: Last performed action.

        Returns:
            Last stocks and cash bias.
        """
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1] - 1
        last_stocks = last_action[:, 1:].reshape((batch_size, 1, stocks, 1))
        cash_bias = last_action[:, 0].reshape((batch_size, 1, 1, 1))
        return last_stocks, cash_bias


class EIIERecurrent(nn.Module):
    def __init__(
        self,
        initial_features: int = 3,
        rec_type: str = "rnn",
        rec_num_layers: int = 20,
        rec_nonlinearity: str = "tanh",
        rec_final_features: int = 20,
        portfolio_size: int = 11,
        device: str = "cpu",
    ) -> EIIERecurrent:
        """Recurrent EIIE (ensemble of identical independent evaluators) policy
        network initializer.

        Args:
            recurrent_type:
            initial_features: Number of input features.
            rec_type: Type of recurrent layers. It can be "rnn" or "lstm".
            rec_num_layers: Number of recurrent layers.
            rec_nonlinearity: Activation function to be used in the recurrent
                units. Can be "relu" or "tanh". Only used if rec_type is
                torch.nn.RNN.
            rec_final_features: Size of final recurrent channels.
            portfolio_size: Number of assets in portfolio.
            device: Device in which the neural network will be run.

        Note:
            Reference article: https://doi.org/10.48550/arXiv.1706.10059.
        """
        super().__init__()
        self.device = device

        self.recurrent_nets = []
        for i in range(portfolio_size):
            if rec_type == "rnn":
                self.recurrent_nets.append(
                    nn.RNN(
                        initial_features,
                        rec_final_features,
                        num_layers=rec_num_layers,
                        nonlinearity=rec_nonlinearity,
                        batch_first=True,
                        device=self.device,
                    )
                )
            else:
                self.recurrent_nets.append(
                    nn.LSTM(
                        initial_features,
                        rec_final_features,
                        num_layers=rec_num_layers,
                        batch_first=True,
                        device=self.device,
                    )
                )
            if self.device != "cpu":
                self.recurrent_nets[i].flatten_parameters()

        # stateless model to be used in functional recurrent call.
        self.base_recurrent_net = copy.deepcopy(self.recurrent_nets[0])

        self.final_convolution = nn.Conv2d(
            in_channels=rec_final_features + 1,
            out_channels=1,
            kernel_size=(1, 1),
            device=self.device,
        )

        self.softmax = nn.Sequential(nn.Softmax(dim=-1))

    def forward(
        self, observation: torch.Tensor, last_action: torch.Tensor
    ) -> torch.Tensor:
        """Policy network's forward propagation. Defines a most favorable
        action of this policy given the inputs.

        Args:
            observation: environment observation.
            last_action: Last action performed by agent.

        Returns:
            Action to be taken.
        """
        last_stocks, cash_bias = self._process_last_action(last_action)
        cash_bias = torch.zeros_like(cash_bias).to(self.device)

        params, buffers = stack_module_state(self.recurrent_nets)
        input = torch.permute(
            observation, (2, 0, 3, 1)
        )  # [portfolio_size, N, time_window, initial_features]

        recurrent_output, _ = torch.vmap(self._functional_recurrent_net)(
            params, buffers, input
        )  # shape [portfolio_size, N, time_window, rec_final_features]

        recurrent_output = torch.permute(
            recurrent_output, (1, 3, 0, 2)
        )  # shape [N, rec_final_features, portfolio_size, time_window]

        recurrent_output = recurrent_output[
            :, :, :, -1
        ]  # # shape [N, rec_final_features, portfolio_size, 1]

        # add last stock weights
        output = torch.cat(
            [last_stocks, recurrent_output], dim=1
        )  # shape [N, rec_final_features + 1, PORTFOLIO_SIZE, 1]
        output = self.final_convolution(output)  # shape [N, 1, PORTFOLIO_SIZE, 1]
        output = torch.cat(
            [cash_bias, output], dim=2
        )  # shape [N, 1, PORTFOLIO_SIZE + 1, 1]

        # output shape must be [N, features] = [N, PORTFOLIO_SIZE + 1], being N batch size (1)
        # and size the number of features (weights vector).
        output = torch.squeeze(output, 3)
        output = torch.squeeze(output, 1)  # shape [N, PORTFOLIO_SIZE + 1]

        output = self.softmax(output)

        return output

    def _process_last_action(
        self, last_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process the last action to retrieve cash bias and last stocks.

        Args:
          last_action: Last performed action.

        Returns:
            Last stocks and cash bias.
        """
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1] - 1
        last_stocks = last_action[:, 1:].reshape((batch_size, 1, stocks, 1))
        cash_bias = last_action[:, 0].reshape((batch_size, 1, 1, 1))
        return last_stocks, cash_bias

    def _functional_recurrent_net(self, params, buffers, x) -> Any:
        """Functional call of a recurrent net.

        Args:
            params: Input model parameters.
            buffers: Input model buffers.
            x: Input data.

        Note:
            Since the code use this function inside a vmap, a weightless version
            of a individual recurrent net is used in the function call.

        Returns:
            The result of calling the module.
        """
        if self.device != "cpu":
            self.base_recurrent_net.flatten_parameters()
        return functional_call(self.base_recurrent_net, (params, buffers), (x,))