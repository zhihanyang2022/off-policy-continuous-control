import gin
import torch
import torch.nn as nn


@gin.configurable(module=__name__)
class MLP(nn.Module):
    """
    A easy-to-use class for creating MLPs.

    Examples (may not appear in practice but are illustrative):

    (1) If state space is 10-dimensional and action space is 3-dimensional, you can create a MLP
        with num_in=10, hidden_dimensions=(256, 256), num_out=3.

    (2) If you want to map state to some embedding space (with no clear interpretation and you would
        like to configure the actual output dimension with gin), you should create a MLP with
        num_in=10, hidden_dimension=(256, 256), num_out=None.
    """

    def __init__(self, num_in, num_out, final_activation, hidden_dimensions=gin.REQUIRED):
        super().__init__()

        tensor_dimensions = [num_in]
        if hidden_dimensions is not None:
            tensor_dimensions.extend(hidden_dimensions)
        if num_out is not None:
            tensor_dimensions.append(num_out)

        num_layers = len(tensor_dimensions) - 1
        layers = []
        input_dimensions, output_dimensions = tensor_dimensions[:-1], tensor_dimensions[1:]
        for i, (input_dimension, output_dimension) in enumerate(zip(input_dimensions, output_dimensions)):
            layers.extend([
                nn.Linear(input_dimension, output_dimension),
                final_activation if i == num_layers - 1 else nn.ReLU(),
            ])
        self.net = nn.Sequential(*layers)

        self.actual_num_out = tensor_dimensions[-1]

    def forward(self, tensor):
        return self.net(tensor)


class MLPTanhActor(nn.Module):
    """Output actions from [-1, 1]."""
    pass


class MLPGaussianActor(nn.Module):
    """Output parameters for some multi-dimensional zero-covariance Gaussian distribution."""

    def __init__(self, input_dim, action_dim):
        super().__init__()

        self.shared_net = MLP(num_in=input_dim, num_out=None, final_activation=nn.ReLU())
        self.means_layer = nn.Linear(in_features=self.shared_net.actual_num_out, out_features=action_dim)
        self.log_stds_layer = nn.Linear(in_features=self.shared_net.actual_num_out, out_features=action_dim)

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
        self.net = MLP(num_in=input_dim + action_dim, num_out=1, final_activation=nn.Identity())

    def forward(self, states: torch.tensor, actions: torch.tensor):
        return self.net(torch.cat([states, actions], dim=1))
