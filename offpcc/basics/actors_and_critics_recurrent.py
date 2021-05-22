import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


# class RecurrentTanhActor(nn.Module):
#     """Output actions from [-1, 1]."""
#     def __init__(self, input_dim, action_dim):
#         super().__init__()
#         self.net = make_MLP(num_in=input_dim, num_out=action_dim, final_activation=nn.Tanh())
#
#     def forward(self, states: torch.tensor):
#         return self.net(states)


class RecurrentGaussianActor(nn.Module):
    """Output parameters for some multi-dimensional zero-covariance Gaussian distribution."""

    def __init__(self, input_dim, action_dim):

        super().__init__()

        self.pre_lstm = nn.Linear(in_features=input_dim, out_features=64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.lstm.flatten_parameters()  # added this to resolve some arbitrary warning

        self.mlp_layer_1 = nn.Linear(in_features=64, out_features=256)
        self.mlp_layer_2 = nn.Linear(in_features=256, out_features=256)

        self.means_layer = nn.Linear(in_features=256, out_features=action_dim)
        self.log_stds_layer = nn.Linear(in_features=256, out_features=action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, observations: torch.tensor) -> tuple:

        x = F.relu(self.pre_lstm(observations))
        x, _ = self.lstm(x)

        x = F.relu(self.mlp_layer_1(x))
        x = F.relu(self.mlp_layer_2(x))

        means, log_stds = self.means_layer(x), self.log_stds_layer(x)
        stds = torch.exp(torch.clamp(log_stds, self.LOG_STD_MIN, self.LOG_STD_MAX))

        return means, stds

    def do_inference(self, observation: torch.tensor, hidden_states: tuple) -> tuple:

        x = F.relu(self.pre_lstm(observation))
        x, hidden_states = self.lstm(x, hidden_states)  # update hidden states

        x = F.relu(self.mlp_layer_1(x))
        x = F.relu(self.mlp_layer_2(x))

        means, log_stds = self.means_layer(x), self.log_stds_layer(x)
        stds = torch.exp(torch.clamp(log_stds, self.LOG_STD_MIN, self.LOG_STD_MAX))

        return means, stds, hidden_states


class RecurrentCritic(nn.Module):

    def __init__(self, input_dim, action_dim):

        super().__init__()

        self.pre_lstm = nn.Linear(in_features=input_dim, out_features=64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.lstm.flatten_parameters()  # added this to resolve some arbitrary warning

        self.mlp_layer_1 = nn.Linear(in_features=64+action_dim, out_features=256)
        self.mlp_layer_2 = nn.Linear(in_features=256, out_features=256)

        self.q_values = nn.Linear(in_features=256, out_features=1)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, observations: torch.tensor, actions: torch.tensor):

        x = F.relu(self.pre_lstm(observations))
        x, _ = self.lstm(x)  # mentally think about the output of lstm as "states"

        x = F.relu(self.mlp_layer_1(torch.cat([x, actions], dim=2)))
        x = F.relu(self.mlp_layer_2(x))

        q_values = self.q_values(x)

        return q_values

