import gin
import torch
import torch.nn as nn

@gin.configurable
def get_net(
        num_in,
        num_out,
        final_activation,
        num_hidden_layers,
        num_neurons_per_hidden_layer
    ) -> nn.Sequential:

    layers = []

    layers.extend([
        nn.Linear(num_in, num_neurons_per_hidden_layer),
        nn.ReLU(),
    ])

    for _ in range(num_hidden_layers):
        layers.extend([
            nn.Linear(num_neurons_per_hidden_layer, num_neurons_per_hidden_layer),
            nn.ReLU(),
        ])

    layers.append(nn.Linear(num_neurons_per_hidden_layer, num_out))

    if final_activation is not None:
        layers.append(final_activation)

    return nn.Sequential(*layers)

class MLPTanhActor(nn.Module):
    """Output actions from [-1, 1]."""
    pass

class MLPGaussianActor(nn.Module):

    """Output continuous actions from some multi-dimensional spherical gaussian distribution."""

    def __init__(self, input_dim, action_dim):
        super().__init__()

        # I don't think means and log_stds should have a shared network at front.
        self.means_net = get_net(num_in=input_dim, num_out=action_dim, final_activation=None)
        self.log_stds_net = get_net(num_in=input_dim, num_out=action_dim, final_activation=None)

    def forward(self, states: torch.tensor) -> tuple:

        means, log_stds = self.means_net(states), self.log_stds_net(states)

        LOG_STD_MAX = 2
        LOG_STD_MIN = -20

        stds = torch.exp(torch.clamp(log_stds, LOG_STD_MIN, LOG_STD_MAX))

        return means, stds

class MLPCritic(nn.Module):

    def __init__(self, input_dim, action_dim):
        pass