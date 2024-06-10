import numpy as np
import torch
from torch import nn


class EIIE(nn.Module):
    def __init__(
        self,
        initial_features=3,
        k_size=3,
        conv_mid_features=2,
        conv_final_features=20,
        time_window=50,
        device="cpu",
    ):
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
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_size),
            ),
            nn.ReLU(),
        )

        self.final_convolution = nn.Conv2d(
            in_channels=conv_final_features + 1, out_channels=1, kernel_size=(1, 1)
        )

        self.softmax = nn.Sequential(nn.Softmax(dim=-1))

    def mu(self, observation, last_action):
        """Defines a most favorable action of this policy given input x.

        Args:
          observation: environment observation.
          last_action: Last action performed by agent.

        Returns:
          Most favorable action.
        """

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        observation = observation.to(self.device).float()

        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action)
        last_action = last_action.to(self.device).float()

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

    def forward(self, observation, last_action):
        """Policy network's forward propagation.

        Args:
          observation: Environment observation (dictionary).
          last_action: Last action performed by the agent.

        Returns:
          Action to be taken (numpy array).
        """
        mu = self.mu(observation, last_action)
        action = mu.cpu().detach().numpy().squeeze()
        return action

    def _process_last_action(self, last_action):
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
        initial_features=3,
        rec_type="rnn",
        rec_num_layers=20,
        rec_nonlinearity="tanh",
        rec_final_features=20,
        portfolio_size=11,
        device="cpu",
    ):
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

        self.final_convolution = nn.Conv2d(
            in_channels=rec_final_features + 1,
            out_channels=1,
            kernel_size=(1, 1),
            device=self.device,
        )

        self.softmax = nn.Sequential(nn.Softmax(dim=-1))

    def mu(self, observation, last_action):
        """Defines a most favorable action of this policy given input x.

        Args:
          observation: environment observation.
          last_action: Last action performed by agent.

        Returns:
          Most favorable action.
        """

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        observation = observation.to(
            self.device
        ).float()  # shape [N, initial_features, num_assets, time_window]

        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action)
        last_action = last_action.to(self.device).float()

        last_stocks, cash_bias = self._process_last_action(last_action)
        cash_bias = torch.zeros_like(cash_bias).to(self.device)

        # run a recurrent net for every asset in portfolio
        recurrent_outputs = []
        for index, net in enumerate(self.recurrent_nets):
            # memory optimization for GPU training
            if self.device != "cpu":
                net.flatten_parameters()
            input = observation[:, :, index, :].transpose(
                1, 2
            )  # shape [N, time_window, initial_features]
            output, _ = net(input)  # shape [N, time_window, rec_final_features]
            output = output[:, -1, :]  # shape [N, rec_final_features]
            output = output.unsqueeze(-1).unsqueeze(
                -1
            )  # shape [N, rec_final_features, 1, 1]
            recurrent_outputs.append(output)

        # concatenate recurrent outputs
        recurrent_outputs = torch.cat(
            recurrent_outputs, dim=2
        )  # shape [N, rec_final_features, PORTFOLIO_SIZE, 1]

        # add last stock weights
        output = torch.cat(
            [last_stocks, recurrent_outputs], dim=1
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

    def forward(self, observation, last_action):
        """Policy network's forward propagation.

        Args:
          observation: Environment observation (dictionary).
          last_action: Last action performed by the agent.

        Returns:
          Action to be taken (numpy array).
        """
        mu = self.mu(observation, last_action)
        action = mu.cpu().detach().numpy().squeeze()
        return action

    def _process_last_action(self, last_action):
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
