from __future__ import annotations

import torch

from torch import nn


class EI3(nn.Module):
    def __init__(
        self,
        initial_features: int = 3,
        k_short: int = 3,
        k_medium: int = 21,
        conv_mid_features: int = 3,
        conv_final_features: int = 20,
        time_window: int = 50,
        device: str = "cpu",
    ) -> EI3:
        """EI3 (ensemble of identical independent inception) policy network
        initializer.

        Args:
            initial_features: Number of input features.
            k_short: Size of short convolutional kernel.
            k_medium: Size of medium convolutional kernel.
            conv_mid_features: Size of intermediate convolutional channels.
            conv_final_features: Size of final convolutional channels.
            time_window: Size of time window used as agent's state.
            device: Device in which the neural network will be run.

        Note:
            Reference article: https://doi.org/10.1145/3357384.3357961.
        """
        super().__init__()
        self.device = device

        n_short = time_window - k_short + 1
        n_medium = time_window - k_medium + 1
        n_long = time_window

        self.short_term = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features,
                out_channels=conv_mid_features,
                kernel_size=(1, k_short),
                device=self.device,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_short),
                device=self.device,
            ),
            nn.ReLU(),
        )

        self.mid_term = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=conv_mid_features,
                kernel_size=(1, k_medium),
                device=self.device,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_medium),
                device=self.device,
            ),
            nn.ReLU(),
        )

        self.long_term = nn.Sequential(nn.MaxPool2d(kernel_size=(1, n_long)), nn.ReLU())

        self.final_convolution = nn.Conv2d(
            in_channels=2 * conv_final_features + initial_features + 1,
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

        short_features = self.short_term(observation)
        medium_features = self.mid_term(observation)
        long_features = self.long_term(observation)

        features = torch.cat(
            [last_stocks, short_features, medium_features, long_features], dim=1
        )
        output = self.final_convolution(features)
        output = torch.cat([cash_bias, output], dim=2)

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
