import os
import gin
import numpy as np

import torch
import torch.optim as optim

from basics.abstract_algorithm import OffPolicyRLAlgorithm
from basics.actors_and_critics import MLPTanhActor, MLPCritic
from basics.replay_buffer import Batch
from basics.cuda_utils import get_device


@gin.configurable(module=__name__)
class TD3(OffPolicyRLAlgorithm):

    """Twin Delayed DDPG"""

    def __init__(
        self,
        input_dim,
        action_dim,
        action_noise=gin.REQUIRED,  # standard deviation of action noise
        target_noise=gin.REQUIRED,  # standard deviation of target smoothing noise
        noise_clip=gin.REQUIRED,    # max abs value of target smoothing noise
        gamma=gin.REQUIRED,
        lr=gin.REQUIRED,
        polyak=gin.REQUIRED,
        policy_delay=gin.REQUIRED
    ):

        # ===== networks =====

        self.actor = MLPTanhActor(input_dim, action_dim).to(get_device())
        self.actor_target = MLPTanhActor(input_dim, action_dim).to(get_device())
        self.actor_target.eval()
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.Q1 = MLPCritic(input_dim, action_dim).to(get_device())
        self.Q1_target = MLPCritic(input_dim, action_dim).to(get_device())
        self.Q1_target.eval()
        self.Q1_target.load_state_dict(self.Q1.state_dict())

        self.Q2 = MLPCritic(input_dim, action_dim).to(get_device())
        self.Q2_target = MLPCritic(input_dim, action_dim).to(get_device())
        self.Q2_target.eval()
        self.Q2_target.load_state_dict(self.Q2.state_dict())

        # ===== optimizers =====

        # ref: https://pytorch.org/docs/stable/optim.html
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=lr)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=lr)

        # ===== hyper-parameters =====

        self.gamma = gamma
        self.polyak = polyak

        self.action_noise = action_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        self.policy_delay = policy_delay
        self.num_Q_updates = 0

    def act(self, state: np.array, deterministic: bool) -> np.array:
        with torch.no_grad():
            state = torch.tensor(state).unsqueeze(0).float().to(get_device())
            greedy_action = self.actor(state).cpu().numpy()[0]
            # use [0] instead of un-squeeze because un-squeeze gets rid of all extra brackets but we need one
            if not deterministic:
                return np.clip(greedy_action + self.action_noise * np.random.randn(len(greedy_action)), -1.0, 1.0)
            else:
                return greedy_action

    @staticmethod
    def clip_gradient(net) -> None:
        for param in net.parameters():
            param.grad.data.clamp_(-1, 1)

    def polyak_update(self, old_net, new_net) -> None:
        for old_param, new_param in zip(old_net.parameters(), new_net.parameters()):
            old_param.data.copy_(old_param.data * self.polyak + new_param.data * (1 - self.polyak))

    def update_networks(self, batch: Batch):

        # ==================================================
        # bellman equation loss (just like Q-learning)
        # ==================================================

        with torch.no_grad():

            na = self.actor_target(batch.ns)
            noise = torch.clip(
                torch.randn(na.size()) * self.target_noise, -self.noise_clip, self.noise_clip
            ).to(get_device())
            smoothed_na = na + noise
            targets = batch.r + \
                      self.gamma * (1 - batch.d) * \
                      torch.min(self.Q1_target(batch.ns, smoothed_na), self.Q2_target(batch.ns, smoothed_na))

        Q1_pred = self.Q1(batch.s, batch.a)
        Q1_loss = torch.mean((Q1_pred - targets) ** 2)

        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.clip_gradient(self.Q1)
        self.Q1_optimizer.step()

        Q2_pred = self.Q2(batch.s, batch.a)
        Q2_loss = torch.mean((Q2_pred - targets) ** 2)

        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.clip_gradient(self.Q2)
        self.Q2_optimizer.step()

        self.num_Q_updates += 1

        # ==================================================
        # policy loss (not present in Q-learning)
        # ==================================================

        if self.num_Q_updates % self.policy_delay == 0:  # delayed policy update; special in TD3

            for param in self.Q1.parameters():
                param.requires_grad = False
            for param in self.Q2.parameters():
                param.requires_grad = False

            a = self.actor(batch.s)
            q1_values = self.Q1(batch.s, a)

            policy_loss = - torch.mean(q1_values)  # minimizing this loss is maximizing the q values

            # ==================================================
            # backpropagation and gradient descent
            # ==================================================

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.clip_gradient(self.actor)
            self.actor_optimizer.step()

            for param in self.Q1.parameters():
                param.requires_grad = True
            for param in self.Q2.parameters():
                param.requires_grad = True

        self.polyak_update(old_net=self.actor_target, new_net=self.actor)
        self.polyak_update(old_net=self.Q1_target, new_net=self.Q1)
        self.polyak_update(old_net=self.Q2_target, new_net=self.Q2)

    def save_actor(self, save_dir: str) -> None:
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor.pth'))

    def load_actor(self, save_dir: str) -> None:
        self.actor.load_state_dict(torch.load(os.path.join(save_dir, 'actor.pth')))