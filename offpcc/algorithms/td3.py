import os
import gin
import numpy as np

import torch
import torch.nn as nn
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

        # networks

        self.actor = MLPTanhActor(input_dim, action_dim).to(get_device())
        self.actor_targ = MLPTanhActor(input_dim, action_dim).to(get_device())
        self.actor_targ.eval()
        self.actor_targ.load_state_dict(self.actor.state_dict())

        self.Q1 = MLPCritic(input_dim, action_dim).to(get_device())
        self.Q1_targ = MLPCritic(input_dim, action_dim).to(get_device())
        self.Q1_targ.eval()
        self.Q1_targ.load_state_dict(self.Q1.state_dict())

        self.Q2 = MLPCritic(input_dim, action_dim).to(get_device())
        self.Q2_targ = MLPCritic(input_dim, action_dim).to(get_device())
        self.Q2_targ.eval()
        self.Q2_targ.load_state_dict(self.Q2.state_dict())

        # optimizers

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=lr)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=lr)

        # hyper-parameters

        self.gamma = gamma
        self.polyak = polyak

        self.action_noise = action_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        self.policy_delay = policy_delay

        # miscellaneous

        self.num_Q_updates = 0  # for delaying updates
        self.action_dim = action_dim  # for shape checking

    def act(self, state: np.array, deterministic: bool) -> np.array:
        with torch.no_grad():
            state = torch.tensor(state).unsqueeze(0).float().to(get_device())
            greedy_action = self.actor(state).view(-1).cpu().numpy()  # view as 1d -> to cpu -> to numpy
            if deterministic:
                return greedy_action
            else:
                return np.clip(greedy_action + self.action_noise * np.random.randn(len(greedy_action)), -1.0, 1.0)

    def polyak_update(self, target_net: nn.Module, prediction_net: nn.Module) -> None:
        for target_param, prediction_param in zip(target_net.parameters(), prediction_net.parameters()):
            target_param.data.copy_(target_param.data * self.polyak + prediction_param.data * (1 - self.polyak))

    def update_networks(self, batch: Batch):

        bs = len(b.ns)  # for shape checking

        # compute prediction

        Q1_pred = self.Q1(batch.s, batch.a)
        Q2_pred = self.Q2(batch.s, batch.a)

        # compute target (n stands for next)

        with torch.no_grad():

            na = self.actor_target(batch.ns)
            noise = torch.clamp(
                torch.randn(na.size()) * self.target_noise, -self.noise_clip, self.noise_clip
            ).to(get_device())
            smoothed_na = torch.clamp(na + noise, -1, 1)

            n_min_Q_targ = torch.min(self.Q1_targ(batch.ns, smoothed_na), self.Q2_targ(batch.ns, smoothed_na))

            targets = batch.r + self.gamma * (1 - batch.d) * n_min_Q_targ

            assert na.shape == (bs, self.action_dim)
            assert n_min_Q_targ.shape == (bs, 1)
            assert targets.shape == (bs, 1)

        # compute td error

        Q1_loss = torch.mean((Q1_pred - targets) ** 2)
        Q2_loss = torch.mean((Q2_pred - targets) ** 2)

        assert Q1_loss.shape == ()
        assert Q2_loss.shape == ()

        # reduce td error

        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.Q1_optimizer.step()

        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.Q2_optimizer.step()

        self.num_Q_updates += 1

        if self.num_Q_updates % self.policy_delay == 0:  # delayed policy update; special in TD3

            for param in self.Q1.parameters():
                param.requires_grad = False
            for param in self.Q2.parameters():
                param.requires_grad = False

            # compute policy loss

            a = self.actor(batch.s)
            Q1_val = self.Q1(batch.s, a)  # val stands for values
            policy_loss = - torch.mean(Q1_val)

            assert a.shape == (bs, self.action_dim)
            assert Q1_val.shape == (bs, 1)
            assert policy_loss.shape == ()

            # reduce policy loss

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            for param in self.Q1.parameters():
                param.requires_grad = True
            for param in self.Q2.parameters():
                param.requires_grad = True

            # # update target networks

            self.polyak_update(target_net=self.actor_targ, prediction_net=self.actor)

            self.polyak_update(target_net=self.Q1_targ, prediction_net=self.Q1)
            self.polyak_update(target_net=self.Q2_targ, prediction_net=self.Q2)

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