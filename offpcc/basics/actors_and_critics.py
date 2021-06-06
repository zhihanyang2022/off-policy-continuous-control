import gin
import torch
import torch.nn as nn


@gin.configurable(module=__name__)
def make_MLP(num_in, num_out, final_activation, hidden_dimensions=(256, 256)):

    tensor_dimensions = [num_in]
    if hidden_dimensions is not None:
        tensor_dimensions.extend(hidden_dimensions)
    if num_out is not None:
        tensor_dimensions.append(num_out)

    num_layers = len(tensor_dimensions) - 1
    layers = []
    input_dimensions, output_dimensions = tensor_dimensions[:-1], tensor_dimensions[1:]
    for i, (input_dimension, output_dimension) in enumerate(zip(input_dimensions, output_dimensions)):
        if i == num_layers - 1:
            if final_activation is None:
                layers.append(nn.Linear(input_dimension, output_dimension))
            else:
                layers.extend([
                    nn.Linear(input_dimension, output_dimension),
                    final_activation,
                ])
        else:
            layers.extend([
                nn.Linear(input_dimension, output_dimension),
                nn.ReLU(),
            ])
    net = nn.Sequential(*layers)

    if num_out is None:
        actual_num_out = tensor_dimensions[-1]
        return net, actual_num_out
    else:
        return net  # actual_num_out would just be num_out


class MLPTanhActor(nn.Module):
    """Output actions from [-1, 1]."""
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = make_MLP(num_in=input_dim, num_out=action_dim, final_activation=nn.Tanh())

    def forward(self, states: torch.tensor):
        return self.net(states)


class MLPGaussianActor(nn.Module):
    """Output parameters for some multi-dimensional zero-covariance Gaussian distribution."""

    def __init__(self, input_dim, action_dim):
        super().__init__()

        self.shared_net, actual_num_out = make_MLP(num_in=input_dim, num_out=None, final_activation=nn.ReLU())
        self.means_layer = nn.Linear(in_features=actual_num_out, out_features=action_dim)
        self.log_stds_layer = nn.Linear(in_features=actual_num_out, out_features=action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, states: torch.tensor) -> tuple:
        out = self.shared_net(states)
        means, log_stds = self.means_layer(out), self.log_stds_layer(out)
        stds = torch.exp(torch.clamp(log_stds, self.LOG_STD_MIN, self.LOG_STD_MAX))
        return means, stds


class MLPCritic(nn.Module):

    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = make_MLP(num_in=input_dim + action_dim, num_out=1, final_activation=None)

    def forward(self, states: torch.tensor, actions: torch.tensor):
        return self.net(torch.cat([states, actions], dim=-1))
