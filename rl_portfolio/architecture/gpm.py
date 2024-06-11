import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import Sequential
from torch_geometric.utils import to_dense_batch


class GPM(nn.Module):
    def __init__(
        self,
        edge_index,
        edge_type,
        nodes_to_select,
        initial_features=3,
        k_short=3,
        k_medium=21,
        conv_mid_features=3,
        conv_final_features=20,
        graph_layers=1,
        time_window=50,
        softmax_temperature=1,
        device="cpu",
    ):
        """GPM (Graph-based Portfolio Management) policy network initializer.

        Args:
            edge_index: Graph connectivity in COO format.
            edge_type: Type of each edge in edge_index.
            nodes_to_select: ID of nodes to be selected to the portfolio.
            initial_features: Number of input features.
            k_short: Size of short convolutional kernel.
            k_medium: Size of medium convolutional kernel.
            conv_mid_features: Size of intermediate convolutional channels.
            conv_final_features: Size of final convolutional channels.
            graph_layers: Number of graph neural network layers.
            time_window: Size of time window used as agent's state.
            softmax_temperature: Temperature parameter to softmax function.
            device: Device in which the neural network will be run.

        Note:
            Reference article: https://doi.org/10.1016/j.neucom.2022.04.105.
        """
        super().__init__()
        self.device = device
        self.softmax_temperature = softmax_temperature

        num_relations = np.unique(edge_type).shape[0]

        if isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index)
        self.edge_index = edge_index.to(self.device).long()

        if isinstance(edge_type, np.ndarray):
            edge_type = torch.from_numpy(edge_type)
        self.edge_type = edge_type.to(self.device).long()

        if isinstance(nodes_to_select, np.ndarray):
            nodes_to_select = torch.from_numpy(nodes_to_select)
        elif isinstance(nodes_to_select, list):
            nodes_to_select = torch.tensor(nodes_to_select)
        self.nodes_to_select = nodes_to_select.to(self.device)

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

        feature_size = 2 * conv_final_features + initial_features

        graph_layers_list = []
        for i in range(graph_layers):
            graph_layers_list += [
                (
                    RGCNConv(feature_size, feature_size, num_relations),
                    "x, edge_index, edge_type -> x",
                ),
                nn.LeakyReLU(),
            ]

        self.gcn = Sequential("x, edge_index, edge_type", graph_layers_list)

        self.final_convolution = nn.Conv2d(
            in_channels=2 * feature_size + 1,
            out_channels=1,
            kernel_size=(1, 1),
            device=self.device,
        )

        self.softmax = nn.Sequential(nn.Softmax(dim=-1))

    def forward(self, observation, last_action):
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

        temporal_features = torch.cat(
            [short_features, medium_features, long_features], dim=1
        )  # shape [N, feature_size, num_stocks, 1]

        # add features to graph
        graph_batch = self._create_graph_batch(temporal_features, self.edge_index)

        # set edge index for the batch
        edge_type = self._create_edge_type_for_batch(graph_batch, self.edge_type)

        # perform graph convolution
        graph_features = self.gcn(
            graph_batch.x, graph_batch.edge_index, edge_type
        )  # shape [N * num_stocks, feature_size]
        graph_features, _ = to_dense_batch(
            graph_features, graph_batch.batch
        )  # shape [N, num_stocks, feature_size]
        graph_features = torch.transpose(
            graph_features, 1, 2
        )  # shape [N, feature_size, num_stocks]
        graph_features = torch.unsqueeze(
            graph_features, 3
        )  # shape [N, feature_size, num_stocks, 1]
        graph_features = graph_features.to(self.device)

        # concatenate graph features and temporal features
        features = torch.cat(
            [temporal_features, graph_features], dim=1
        )  # shape [N, 2 * feature_size, num_stocks, 1]

        # perform selection and add last stocks
        features = torch.index_select(
            features, dim=2, index=self.nodes_to_select
        )  # shape [N, 2 * feature_size, portfolio_size, 1]
        features = torch.cat([last_stocks, features], dim=1)

        # final convolution
        output = self.final_convolution(features)  # shape [N, 1, portfolio_size, 1]
        output = torch.cat(
            [cash_bias, output], dim=2
        )  # shape [N, 1, portfolio_size + 1, 1]

        # output shape must be [N, portfolio_size + 1] = [1, portfolio_size + 1], being N batch size
        output = torch.squeeze(output, 3)
        output = torch.squeeze(output, 1)  # shape [N, portfolio_size + 1]

        output = self.softmax(output / self.softmax_temperature)

        return output

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

    def _create_graph_batch(self, features, edge_index):
        """Create a batch of graphs with the features.

        Args:
          features: Tensor of shape [batch_size, feature_size, num_stocks, 1].
          edge_index: Graph connectivity in COO format.

        Returns:
          A batch of graphs with temporal features associated with each node.
        """
        batch_size = features.shape[0]
        graphs = []
        for i in range(batch_size):
            x = features[i, :, :, 0]  # shape [feature_size, num_stocks]
            x = torch.transpose(x, 0, 1)  # shape [num_stocks, feature_size]
            new_graph = Data(x=x, edge_index=edge_index).to(self.device)
            graphs.append(new_graph)
        return Batch.from_data_list(graphs)

    def _create_edge_type_for_batch(self, batch, edge_type):
        """Create the edge type tensor for a batch of graphs.

        Args:
          batch: Batch of graph data.
          edge_type: Original edge type tensor.

        Returns:
          Edge type tensor adapted for the batch.
        """
        batch_edge_type = torch.clone(edge_type).detach()
        for i in range(1, batch.batch_size):
            batch_edge_type = torch.cat(
                [batch_edge_type, torch.clone(edge_type).detach()]
            )
        return batch_edge_type
