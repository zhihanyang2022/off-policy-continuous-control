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

        # self.pre_lstm = nn.Linear(in_features=input_dim, out_features=64)
        self.layer1 = nn.LSTM(input_size=input_dim, hidden_size=256, batch_first=True)
        self.layer2 = nn.Linear(in_features=256, out_features=256)

        self.means_layer = nn.Linear(in_features=256, out_features=action_dim)
        self.log_stds_layer = nn.Linear(in_features=256, out_features=action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, observations: torch.tensor) -> tuple:

        # x = F.relu(self.pre_lstm(observations))

        self.layer1.flatten_parameters()

        x, _ = self.layer1(observations)
        x = F.relu(self.layer2(x))

        means, log_stds = self.means_layer(x), self.log_stds_layer(x)
        stds = torch.exp(torch.clamp(log_stds, self.LOG_STD_MIN, self.LOG_STD_MAX))

        return means, stds

    def do_inference(self, observation: torch.tensor, hidden_states: tuple) -> tuple:

        self.layer1.flatten_parameters()

        # x = F.relu(self.pre_lstm(observation))
        x, hidden_states = self.layer1(observation, hidden_states)  # update hidden states
        x = F.relu(self.layer2(x))

        means, log_stds = self.means_layer(x), self.log_stds_layer(x)
        stds = torch.exp(torch.clamp(log_stds, self.LOG_STD_MIN, self.LOG_STD_MAX))

        return means, stds, hidden_states


class RecurrentCritic(nn.Module):

    def __init__(self, input_dim, action_dim):

        super().__init__()

        # self.pre_lstm = nn.Linear(in_features=input_dim, out_features=64)
        self.layer1 = nn.LSTM(input_size=input_dim, hidden_size=256, batch_first=True)
        self.layer2 = nn.Linear(in_features=256+action_dim, out_features=256)
        self.q_values = nn.Linear(in_features=256, out_features=1)

    def forward(self, observations: torch.tensor, actions: torch.tensor):

        self.layer1.flatten_parameters()

        # x = F.relu(self.pre_lstm(observations))
        x, _ = self.layer1(observations)
        x = F.relu(self.layer2(torch.cat([x, actions], dim=2)))

        q_values = self.q_values(x)

        return q_values

