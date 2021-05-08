import os
import gin
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from basics.abstract_algorithms import OffPolicyRLAlgorithm
from basics.actors_and_critics import MLPGaussianActor, MLPCritic
from basics.buffer import Batch
from basics.utils import clip_gradient

@gin.configurable(module=__name__)
class SAC(OffPolicyRLAlgorithm):

    def __init__(
        self,
        input_dim,
        action_dim,
        gamma,
        alpha,
        lr,
        polyak
    ):

        self.Normal = MLPGaussianActor(input_dim=input_dim, action_dim=action_dim)
        self.Normal_optimizer = optim.Adam(self.Normal.parameters(), lr=lr)

        self.Q1 = MLPCritic(input_dim=input_dim, action_dim=action_dim)
        self.Q1_targ = MLPCritic(input_dim=input_dim, action_dim=action_dim)
        self.Q1_targ.load_state_dict(self.Q1.state_dict())
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=1e-3)

        self.Q2 = MLPCritic(input_dim=input_dim, action_dim=action_dim)
        self.Q2_targ = MLPCritic(input_dim=input_dim, action_dim=action_dim)
        self.Q2_targ.load_state_dict(self.Q2.state_dict())
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=1e-3)

        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak

    def sample_action_and_compute_log_pi(self, state: torch.tensor, use_reparametrization_trick: bool) -> tuple:
        mu_given_s = self.Normal(state)  # in paper, mu represents the normal distribution
        # in paper, u represents the un-squashed action; nu stands for next u's
        # actually, we can just use reparametrization trick in both Step 12 and 14, but it might be good to separate
        # the two cases for pedagogical purposes, i.e., using reparametrization trick is a must in Step 14
        u = mu_given_s.rsample() if use_reparametrization_trick else mu_given_s.sample()
        a = torch.tanh(u)
        # the following line of code is not numerically stable:
        # log_pi_a_given_s = mu_given_s.log_prob(u) - torch.sum(torch.log(1 - torch.tanh(u) ** 2), dim=1)
        # ref: https://github.com/vitchyr/rlkit/blob/0073d73235d7b4265cd9abe1683b30786d863ffe/rlkit/torch/distributions.py#L358
        # ref: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73
        log_pi_a_given_s = mu_given_s.log_prob(u) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(dim=1)
        return a, log_pi_a_given_s

    def polyak_update(self, old_net: nn.Module, new_net: nn.Module) -> None:
        with torch.no_grad():
            for old_param, new_param in zip(old_net.parameters(), new_net.parameters()):
                old_param.data.copy_(old_param.data * self.polyak + new_param.data * (1 - self.polyak))

    def act(self, state: np.array) -> np.array:
        # TODO: add eval mode without noise
        state = torch.tensor(state).unsqueeze(0).float()
        action, _ = self.sample_action_and_compute_log_pi(state, use_reparametrization_trick=False)
        return action.numpy()[0]  # no need to detach first because we are not using the reparametrization trick

    def update_networks(self, b: Batch) -> None:
        # ========================================
        # Step 12: calculating targets
        # ========================================

        with torch.no_grad():

            na, log_pi_na_given_ns = self.sample_action_and_compute_log_pi(b.ns, use_reparametrization_trick=False)
            targets = b.r + self.gamma * (1 - b.d) * \
                      (torch.min(self.Q1_targ(b.ns, na), self.Q2_targ(b.ns, na)) - self.alpha * log_pi_na_given_ns)

        # ========================================
        # Step 13: learning the Q functions
        # ========================================

        Q1_predictions = self.Q1(b.s, b.a)
        Q1_loss = torch.mean((Q1_predictions - targets) ** 2)

        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        clip_gradient(net=self.Q1)
        self.Q1_optimizer.step()

        Q2_predictions = self.Q2(b.s, b.a)
        Q2_loss = torch.mean((Q2_predictions - targets) ** 2)

        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        clip_gradient(net=self.Q2)
        self.Q2_optimizer.step()

        # ========================================
        # Step 14: learning the policy
        # ========================================

        for param in self.Q1.parameters():
            param.requires_grad = False
        for param in self.Q2.parameters():
            param.requires_grad = False

        a, log_pi_a_given_s = self.sample_action_and_compute_log_pi(b.s, use_reparametrization_trick=True)
        policy_loss = - torch.mean(torch.min(self.Q1(b.s, a), self.Q2(b.s, a)) - self.alpha * log_pi_a_given_s)

        self.Normal_optimizer.zero_grad()
        policy_loss.backward()
        clip_gradient(net=self.Normal)
        self.Normal_optimizer.step()

        for param in self.Q1.parameters():
            param.requires_grad = True
        for param in self.Q2.parameters():
            param.requires_grad = True

        # ========================================
        # Step 15: update target networks
        # ========================================

        self.polyak_update(old_net=self.Q1_targ, new_net=self.Q1)
        self.polyak_update(old_net=self.Q2_targ, new_net=self.Q2)

    def save_actor(self, save_dir: str, save_filename: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.Normal.state_dict(), os.path.join(save_dir, save_filename))

    def load_actor(self, save_dir: str, save_filename: str) -> None:
        self.Normal.load_state_dict(torch.load(os.path.join(save_dir, save_filename)))