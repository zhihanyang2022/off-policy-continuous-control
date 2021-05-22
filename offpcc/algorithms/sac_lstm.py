from typing import Union
import gin
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Independent
from torch.nn import SmoothL1Loss

from basics.abstract_algorithm import OffPolicyRLAlgorithm
from basics.actors_and_critics_recurrent import RecurrentGaussianActor, RecurrentCritic
from basics.replay_buffer_recurrent import RecurrentBatch
from basics.cuda_utils import get_device


Q_loss_fn = SmoothL1Loss(reduction='none')

def rescale_loss(loss: torch.tensor, mask: torch.tensor) -> torch.tensor:
    return loss / mask.sum() * np.prod(mask.shape)


@gin.configurable(module=__name__)
class SAC_LSTM(OffPolicyRLAlgorithm):

    """Soft actor-critic with LSTM-based recurrent actor and critic."""

    def __init__(
            self,
            input_dim: int,  # here this means o_dim
            action_dim: int,
            gamma: float = gin.REQUIRED,
            alpha: float = gin.REQUIRED,
            lr: float = gin.REQUIRED,
            polyak: float = gin.REQUIRED
    ):

        # networks

        self.actor = RecurrentGaussianActor(input_dim=input_dim, action_dim=action_dim).to(get_device())

        self.Q1 = RecurrentCritic(input_dim=input_dim, action_dim=action_dim).to(get_device())
        self.Q1_targ = RecurrentCritic(input_dim=input_dim, action_dim=action_dim).to(get_device())
        self.Q1_targ.eval()
        self.Q1_targ.load_state_dict(self.Q1.state_dict())

        self.Q2 = RecurrentCritic(input_dim=input_dim, action_dim=action_dim).to(get_device())
        self.Q2_targ = RecurrentCritic(input_dim=input_dim, action_dim=action_dim).to(get_device())
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

        self.hidden = None

    def restart(self) -> None:
        self.hidden = (torch.zeros(1, 1, 256).to(get_device()), torch.zeros(1, 1, 256).to(get_device()))

    def sample_action_from_distribution(
            self,
            observation: torch.tensor,
            deterministic: bool,
            return_log_prob: bool,
            track_hidden: bool
    ) -> Union[torch.tensor, tuple]:  # tuple of 2 tensors if return_log_prob is True; else torch.tensor

        if track_hidden:  # for online rollout (not for learning)
            means, stds, self.hidden = self.actor.do_inference(observation, self.hidden)
        else:
            means, stds = self.actor(observation)

        bs, num_bptt = means.shape[0], means.shape[1]
        means, stds = means.view(bs * num_bptt, self.action_dim), stds.view(bs * num_bptt, self.action_dim)

        if deterministic:
            u = means
        else:
            mu_given_s = Independent(Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)  # normal distribution
            u = mu_given_s.rsample()

        a = torch.tanh(u).view(bs, num_bptt, self.action_dim)  # shape checking

        if return_log_prob:
            log_pi_a_given_s = mu_given_s.log_prob(u) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(dim=1)
            return a, log_pi_a_given_s.view(bs, num_bptt, 1)  # add another dim to match Q values
        else:
            return a

    def act(self, observation: np.array, deterministic: bool) -> np.array:
        with torch.no_grad():
            observation = torch.tensor(observation).unsqueeze(0).unsqueeze(0).float().to(get_device())
            action = self.sample_action_from_distribution(
                observation,
                deterministic=deterministic,
                return_log_prob=False,
                track_hidden=True
            )
            return action.view(-1).cpu().numpy()  # view as 1d -> to cpu -> to numpy

    def polyak_update(self, target_net: nn.Module, prediction_net: nn.Module) -> None:
        for target_param, prediction_param in zip(target_net.parameters(), prediction_net.parameters()):
            target_param.data.copy_(target_param.data * self.polyak + prediction_param.data * (1 - self.polyak))

    @staticmethod
    def burn_in(tensor):
        return tensor[:, 2:, :]

    def update_networks(self, b: RecurrentBatch) -> dict:

        bs, num_bptt = b.r.shape[0], b.r.shape[1]

        # compute prediction

        Q1_predictions = self.Q1(b.o, b.a)
        Q2_predictions = self.Q2(b.o, b.a)

        assert Q1_predictions.shape == (bs, num_bptt, 1)
        assert Q2_predictions.shape == (bs, num_bptt, 1)

        # compute target (n stands for next)

        with torch.no_grad():

            na, log_pi_na_given_ns = self.sample_action_from_distribution(b.no,
                                                                          deterministic=False,
                                                                          return_log_prob=True,
                                                                          track_hidden=False)

            n_min_Q_targ = torch.min(self.Q1_targ(b.no, na), self.Q2_targ(b.no, na))
            n_entropy = - log_pi_na_given_ns

            targets = b.r + self.gamma * (1 - b.d) * (n_min_Q_targ + self.alpha * n_entropy)

            assert na.shape == (bs, num_bptt, self.action_dim)
            assert log_pi_na_given_ns.shape == (bs, num_bptt, 1)
            assert n_min_Q_targ.shape == (bs, num_bptt, 1)
            assert targets.shape == (bs, num_bptt, 1)

        # burn-in

        Q1_predictions = self.burn_in(Q1_predictions)
        Q2_predictions = self.burn_in(Q2_predictions)
        targets = self.burn_in(targets)
        m = self.burn_in(b.m)

        # compute td error

        Q1_loss_elementwise = Q_loss_fn(Q1_predictions, targets)
        Q1_loss = rescale_loss(torch.mean(m * Q1_loss_elementwise), m)

        Q2_loss_elementwise = Q_loss_fn(Q2_predictions, targets)
        Q2_loss = rescale_loss(torch.mean(m * Q2_loss_elementwise), m)

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

        a, log_pi_a_given_s = self.sample_action_from_distribution(b.o,
                                                                   deterministic=False,
                                                                   return_log_prob=True,
                                                                   track_hidden=False)

        min_Q = torch.min(self.Q1(b.o, a), self.Q2(b.o, a))
        entropy = - log_pi_a_given_s

        min_Q = self.burn_in(min_Q)
        entropy = self.burn_in(entropy)

        policy_loss_elementwise = - (min_Q + self.alpha * entropy)
        policy_loss = rescale_loss(torch.mean(m * policy_loss_elementwise), m)

        assert a.shape == (bs, num_bptt, self.action_dim)
        assert log_pi_a_given_s.shape == (bs, num_bptt - 2, 1)
        assert min_Q.shape == (bs, num_bptt - 2, 1)
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
            '(actor) entropy (sample)': float(entropy.mean()),
            '(actor) policy loss': float(policy_loss)
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
