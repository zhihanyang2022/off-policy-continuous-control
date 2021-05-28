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
from basics.actors_and_critics import MLPGaussianActor, MLPCritic, set_requires_grad_flag
from basics.replay_buffer import Batch
from basics.cuda_utils import get_device


@gin.configurable(module=__name__)
class SAC(OffPolicyRLAlgorithm):

    """
    Soft actor-critic

    The autotuning of the entropy coefficient (alpha) follows almost EXACTLY from SB3's SAC implementation, while
    other code follows from spinup's implementation.
    """

    def __init__(
            self,
            input_dim: int,
            action_dim: int,
            gamma: float = gin.REQUIRED,
            alpha: float = gin.REQUIRED,
            autotune_alpha: bool = gin.REQUIRED,
            lr: float = gin.REQUIRED,
            polyak: float = gin.REQUIRED
    ):

        # networks

        self.actor = MLPGaussianActor(input_dim=input_dim, action_dim=action_dim).to(get_device())

        self.Q1 = MLPCritic(input_dim=input_dim, action_dim=action_dim).to(get_device())
        self.Q1_targ = MLPCritic(input_dim=input_dim, action_dim=action_dim).to(get_device())
        set_requires_grad_flag(self.Q1_targ, False)
        self.Q1_targ.load_state_dict(self.Q1.state_dict())

        self.Q2 = MLPCritic(input_dim=input_dim, action_dim=action_dim).to(get_device())
        self.Q2_targ = MLPCritic(input_dim=input_dim, action_dim=action_dim).to(get_device())
        set_requires_grad_flag(self.Q2_targ, False)
        self.Q2_targ.load_state_dict(self.Q2.state_dict())

        # optimizers

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=lr)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=lr)

        # hyper-parameters

        self.gamma = gamma
        self.autotune_alpha = autotune_alpha
        self.polyak = polyak

        if autotune_alpha:
            self.log_alpha = torch.log(torch.ones(1) * alpha).requires_grad(True).to(get_device())
            self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

        # miscellaneous

        self.input_dim = input_dim
        self.action_dim = action_dim  # for shape checking
        self.target_entropy = - self.action_dim  # int, but it will get broadcasted over a FloatTensor as a float

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

    def get_current_alpha(self):
        if self.autotune_alpha:
            return np.exp(float(self.log_alpha))
        else:
            return self.alpha

    def polyak_update(self, target_net: nn.Module, prediction_net: nn.Module) -> None:
        for target_param, prediction_param in zip(target_net.parameters(), prediction_net.parameters()):
            target_param.data.copy_(target_param.data * self.polyak + prediction_param.data * (1 - self.polyak))

    def update_networks(self, b: Batch) -> dict:

        bs = len(b.ns)  # for shape checking

        # compute prediction

        Q1_predictions = self.Q1(b.s, b.a)
        Q2_predictions = self.Q2(b.s, b.a)

        assert Q1_predictions.shape == (bs, 1)
        assert Q2_predictions.shape == (bs, 1)

        # compute target (n stands for next)

        with torch.no_grad():

            na, log_pi_na_given_ns = self.sample_action_from_distribution(b.ns, deterministic=False,
                                                                          return_log_prob=True)

            n_min_Q_targ = torch.min(self.Q1_targ(b.ns, na), self.Q2_targ(b.ns, na))
            n_sample_entropy = - log_pi_na_given_ns

            targets = b.r + self.gamma * (1 - b.d) * (n_min_Q_targ + self.get_current_alpha() * n_sample_entropy)

            assert na.shape == (bs, self.action_dim)
            assert log_pi_na_given_ns.shape == (bs, 1)
            assert n_min_Q_targ.shape == (bs, 1)
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

        # compute policy loss

        set_requires_grad_flag(self.Q1, False)
        set_requires_grad_flag(self.Q2, False)

        a, log_pi_a_given_s = self.sample_action_from_distribution(b.s, deterministic=False, return_log_prob=True)

        min_Q = torch.min(self.Q1(b.s, a), self.Q2(b.s, a))
        sample_entropy = - log_pi_a_given_s

        policy_loss = - torch.mean(min_Q + self.get_current_alpha() * sample_entropy)

        assert a.shape == (bs, self.action_dim)
        assert log_pi_a_given_s.shape == (bs, 1)
        assert min_Q.shape == (bs, 1)
        assert policy_loss.shape == ()

        # reduce policy loss

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        set_requires_grad_flag(self.Q1, True)
        set_requires_grad_flag(self.Q2, True)

        if self.autotune_alpha:

            # compute log alpha loss

            # derivation to make things more intuitive
            #
            # alpha_loss = - self.log_alpha * (log_pi_a_given_s.detach() + self.target_entropy)
            #            = self.log_alpha * (- log_pi_a_given_s.detach() - self.target_entropy)
            #            = self.log_alpha * (sample_entropy - self.target_entropy)
            #            = self.log_alpha * excess_entropy
            #
            # If excess_entropy > 0, then log_alpha needs to be decreased to reduce alpha_loss, which corresponds to
            # less weighting on sample_entropy in policy loss and hence reduces excess_entropy.
            #
            # If excess_entropy < 0, then log_alpha needs to be increased to reduce alpha_loss, which corresponds to
            # more weighting on sample_entropy in policy loss and hence increases excess_entropy.

            excess_entropy = sample_entropy.detach() - self.target_entropy
            log_alpha_loss = self.log_alpha * excess_entropy

            # reduce log alpha loss

            self.log_alpha_optimizer.zero_grad()
            log_alpha_loss.backward()
            self.log_alpha_optimizer.step()

        else:

            log_alpha_loss = 0

        # update target networks

        self.polyak_update(target_net=self.Q1_targ, prediction_net=self.Q1)
        self.polyak_update(target_net=self.Q2_targ, prediction_net=self.Q2)

        return {
            # for learning the q functions
            '(qfunc) Q1 pred': float(Q1_predictions.mean()),
            '(qfunc) Q2 pred': float(Q2_predictions.mean()),
            '(qfunc) Q1 loss': float(Q1_loss),
            '(qfunc) Q2 loss': float(Q2_loss),
            # for learning the actor
            '(actor) min Q pred': float(min_Q.mean()),
            '(actor) entropy (sample)': float(sample_entropy.mean()),
            '(actor) policy loss': float(policy_loss),
            # for learning the entropy coefficient (alpha)
            '(alpha) alpha': self.get_current_alpha(),
            '(alpha) log alpha loss': float(log_alpha_loss.mean())
        }

    def save_networks(self, save_dir: str) -> None:
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor.pth'))
        torch.save(self.Q1.state_dict(), os.path.join(save_dir, 'Q1.pth'))
        torch.save(self.Q1_targ.state_dict(), os.path.join(save_dir, 'Q1_targ.pth'))
        torch.save(self.Q2.state_dict(), os.path.join(save_dir, 'Q2.pth'))
        torch.save(self.Q2_targ.state_dict(), os.path.join(save_dir, 'Q2_targ.pth'))

    def load_actor(self, save_dir: str) -> None:
        self.actor.load_state_dict(
            torch.load(os.path.join(save_dir, 'actor.pth'), map_location=torch.device(get_device())))

    def load_networks(self, save_dir: str) -> None:
        self.actor.load_state_dict(
            torch.load(os.path.join(save_dir, 'actor.pth'), map_location=torch.device(get_device())))
        self.Q1.load_state_dict(
            torch.load(os.path.join(save_dir, 'Q1.pth'), map_location=torch.device(get_device())))
        self.Q1_targ.load_state_dict(
            torch.load(os.path.join(save_dir, 'Q1_targ.pth'), map_location=torch.device(get_device())))
        self.Q2.load_state_dict(
            torch.load(os.path.join(save_dir, 'Q2.pth'), map_location=torch.device(get_device())))
        self.Q2_targ.load_state_dict(
            torch.load(os.path.join(save_dir, 'Q2_targ.pth'), map_location=torch.device(get_device())))
