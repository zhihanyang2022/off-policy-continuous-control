from typing import Union
import gin
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Independent

from basics.abstract_algorithm import OffPolicyRLAlgorithm
from basics.actors_and_critics import MLPGaussianActor, MLPCritic
from basics.replay_buffer import Batch
from basics.cuda_utils import get_device


@gin.configurable(module=__name__)
class SAC(OffPolicyRLAlgorithm):
    """Soft actor-critic"""

    def __init__(
            self,
            input_dim: int,
            action_dim: int,
            gamma: float = gin.REQUIRED,
            alpha: float = gin.REQUIRED,
            lr: float = gin.REQUIRED,
            polyak: float = gin.REQUIRED
    ):

        # networks

        self.actor = MLPGaussianActor(input_dim=input_dim, action_dim=action_dim).to(get_device())

        self.Q1 = MLPCritic(input_dim=input_dim, action_dim=action_dim).to(get_device())
        self.Q1_targ = MLPCritic(input_dim=input_dim, action_dim=action_dim).to(get_device())
        self.Q1_targ.eval()
        self.Q1_targ.load_state_dict(self.Q1.state_dict())

        self.Q2 = MLPCritic(input_dim=input_dim, action_dim=action_dim).to(get_device())
        self.Q2_targ = MLPCritic(input_dim=input_dim, action_dim=action_dim).to(get_device())
        self.Q2_targ.eval()
        self.Q2_targ.load_state_dict(self.Q2.state_dict())

        # optimizers

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=lr)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=lr)

        # hyper-parameters

        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak

        # miscellaneous

        self.action_dim = action_dim  # for shape checking

    def sample_action_from_distribution(
            self,
            state: torch.tensor,
            deterministic: bool,
            return_log_prob: bool
    ) -> Union[torch.tensor, tuple]:  # tuple of 2 tensors if return_log_prob is True; else torch.tensor

        # notation (from SAC paper):
        # - mu represents the normal distribution
        # - u represents the un-squashed action; nu stands for next u's

        # reference on log_prob computation code:
        # the following line of code (from SAC paper) is not numerically stable:
        # log_pi_a_given_s = mu_given_s.log_prob(u) - torch.sum(torch.log(1 - torch.tanh(u) ** 2), dim=1)
        # the alternative and equivalent way used in this code is copied from:
        # github.com/vitchyr/rlkit/blob/0073d73235d7b4265cd9abe1683b30786d863ffe/rlkit/torch/distributions.py#L358
        # github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73

        means, stds = self.actor(state)

        if deterministic:
            u = means
        else:
            mu_given_s = Independent(Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)
            u = mu_given_s.rsample()

        a = torch.tanh(u).view(-1, self.action_dim)  # shape checking

        if return_log_prob:
            log_pi_a_given_s = mu_given_s.log_prob(u) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(dim=1)
            return a, log_pi_a_given_s.view(-1, 1)  # add another dim to match Q values
        else:
            return a

    def act(self, state: np.array, deterministic: bool) -> np.array:
        with torch.no_grad():
            state = torch.tensor(state).unsqueeze(0).float().to(get_device())
            action = self.sample_action_from_distribution(state, deterministic=deterministic, return_log_prob=False)
            return action.view(-1).cpu().numpy()  # view as 1d -> to cpu -> to numpy

    def polyak_update(self, old_net: nn.Module, new_net: nn.Module) -> None:
        for old_param, new_param in zip(old_net.parameters(), new_net.parameters()):
            old_param.data.copy_(old_param.data * self.polyak + new_param.data * (1 - self.polyak))

    def update_networks(self, b: Batch) -> None:

        bs = len(b.ns)  # for shape checking

        # compute prediction

        Q1_predictions = self.Q1(b.s, b.a)
        Q2_predictions = self.Q2(b.s, b.a)

        assert Q1_predictions.shape == (bs, 1)
        assert Q2_predictions.shape == (bs, 1)

        # compute target

        with torch.no_grad():

            na, log_pi_na_given_ns = self.sample_action_from_distribution(b.ns, deterministic=False,
                                                                          return_log_prob=True)

            min_Q_targ = torch.min(self.Q1_targ(b.ns, na), self.Q2_targ(b.ns, na))
            entropy = - log_pi_na_given_ns

            targets = b.r + self.gamma * (1 - b.d) * (min_Q_targ + self.alpha * entropy)

            assert na.shape == (bs, self.action_dim)
            assert log_pi_na_given_ns.shape == (bs, 1)
            assert min_Q_targ.shape == (bs, 1)
            assert targets.shape == (bs, 1)

        # compute td error

        Q1_loss = torch.mean((Q1_predictions - targets) ** 2)
        Q2_loss = torch.mean((Q2_predictions - targets) ** 2)

        assert Q1_loss.shape == ()
        assert Q2_loss.shape == ()

        # reduce td error

        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.Q1_optimizer.step()

        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.Q2_optimizer.step()

        for param in self.Q1.parameters():
            param.requires_grad = False
        for param in self.Q2.parameters():
            param.requires_grad = False

        # compute policy loss

        a, log_pi_a_given_s = self.sample_action_from_distribution(b.s, deterministic=False, return_log_prob=True)

        min_Q = torch.min(self.Q1(b.s, a), self.Q2(b.s, a))
        entropy = - log_pi_a_given_s

        policy_loss = - torch.mean(min_Q + self.alpha * entropy)

        assert a.shape == (bs, self.action_dim)
        assert log_pi_a_given_s.shape == (bs, 1)
        assert min_Q.shape == (bs, 1)
        assert policy_loss.shape == ()

        # reduce policy loss

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        for param in self.Q1.parameters():
            param.requires_grad = True
        for param in self.Q2.parameters():
            param.requires_grad = True

        # update target networks

        self.polyak_update(old_net=self.Q1_targ, new_net=self.Q1)
        self.polyak_update(old_net=self.Q2_targ, new_net=self.Q2)

    def save_actor(self, save_dir: str) -> None:
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor.pth'))

    def load_actor(self, save_dir: str) -> None:
        self.actor.load_state_dict(
            torch.load(os.path.join(save_dir, 'actor.pth'), map_location=torch.device(get_device())))
