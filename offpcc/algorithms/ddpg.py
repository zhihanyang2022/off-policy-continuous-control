import os
import gin
import numpy as np

import torch
import torch.optim as optim

from basics.abstract_algorithm import OffPolicyRLAlgorithm
from basics.actors_and_critics import MLPTanhActor, MLPCritic, set_requires_grad_flag
from basics.replay_buffer import Batch
from basics.cuda_utils import get_device


@gin.configurable(module=__name__)
class DDPG(OffPolicyRLAlgorithm):

    """Deep deterministic policy gradient"""

    def __init__(
            self,
            input_dim,
            action_dim,
            action_noise=gin.REQUIRED,
            gamma=gin.REQUIRED,
            lr=gin.REQUIRED,
            polyak=gin.REQUIRED,
        ):

        # networks

        self.actor = MLPTanhActor(input_dim, action_dim).to(get_device())
        self.actor_targ = MLPTanhActor(input_dim, action_dim).to(get_device())
        set_requires_grad_flag(self.actor_targ, False)
        self.actor_targ.load_state_dict(self.actor.state_dict())

        self.Q = MLPCritic(input_dim, action_dim).to(get_device())
        self.Q_targ = MLPCritic(input_dim, action_dim).to(get_device())
        set_requires_grad_flag(self.Q_targ, False)
        self.Q_targ.load_state_dict(self.Q.state_dict())

        # optimizers

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        # hyper-parameters

        self.gamma = gamma
        self.action_noise = action_noise
        self.polyak = polyak

        # miscellaneous

        self.action_dim = action_dim

    def act(self, state: np.array, deterministic: bool) -> np.array:
        with torch.no_grad():
            state = torch.tensor(state).unsqueeze(0).float().to(get_device())
            greedy_action = self.actor(state).view(-1).cpu().numpy()  # view as 1d -> to cpu -> to numpy
            if not deterministic:
                return np.clip(greedy_action + self.action_noise * np.random.randn(len(greedy_action)), -1.0, 1.0)
            else:
                return greedy_action

    def polyak_update(self, old_net, new_net) -> None:
        for old_param, new_param in zip(old_net.parameters(), new_net.parameters()):
            old_param.data.copy_(old_param.data * self.polyak + new_param.data * (1 - self.polyak))

    def update_networks(self, b: Batch):

        bs = len(b.ns)  # for shape checking

        # compute predictions

        predictions = self.Q(b.s, b.a)

        assert predictions.shape == (bs, 1)

        # compute targets

        with torch.no_grad():

            na = self.actor_targ(b.ns)
            targets = b.r + self.gamma * (1 - b.d) * self.Q_targ(b.ns, na)

            assert na.shape == (bs, self.action_dim)
            assert targets.shape == (bs, 1)

        # compute td error

        Q_loss = torch.mean((predictions - targets.detach()) ** 2)

        assert Q_loss.shape == ()

        # reduce td error

        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # compute policy loss

        set_requires_grad_flag(self.Q, False)

        a = self.actor(b.s)
        Q_values = self.Q(b.s, a)
        policy_loss = - torch.mean(Q_values)

        assert a.shape == (bs, self.action_dim)
        assert policy_loss.shape == ()

        # reduce policy loss

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        set_requires_grad_flag(self.Q, True)

        # update target networks

        self.polyak_update(old_net=self.actor_targ, new_net=self.actor)
        self.polyak_update(old_net=self.Q_targ, new_net=self.Q)

        return {}

    def save_networks(self, save_dir: str) -> None:
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor.pth'))
        torch.save(self.Q.state_dict(), os.path.join(save_dir, 'Q.pth'))
        torch.save(self.Q_targ.state_dict(), os.path.join(save_dir, 'Q_targ.pth'))

    def load_actor(self, save_dir: str) -> None:
        self.actor.load_state_dict(
            torch.load(os.path.join(save_dir, 'actor.pth'), map_location=torch.device(get_device())))

    def load_networks(self, save_dir: str) -> None:
        self.actor.load_state_dict(
            torch.load(os.path.join(save_dir, 'actor.pth'), map_location=torch.device(get_device())))
        self.Q1.load_state_dict(
            torch.load(os.path.join(save_dir, 'Q1.pth'), map_location=torch.device(get_device())))
        self.Q1_target.load_state_dict(
            torch.load(os.path.join(save_dir, 'Q1_targ.pth'), map_location=torch.device(get_device())))
