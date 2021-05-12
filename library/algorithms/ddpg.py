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
class DDPG(OffPolicyRLAlgorithm):

    def __init__(
            self,
            input_dim,
            action_dim,
            action_noise=gin.REQUIRED,
            gamma=gin.REQUIRED,
            lr=gin.REQUIRED,
            polyak=gin.REQUIRED,
        ):

        # ===== networks =====

        # Q: (s, a_) --- network --> scalar
        # Q_target    : (s, a_) --- network --> scalar
        # actor: s --- network --> a_ (in (0, 1) and hence need to undo normalization)

        self.actor = MLPTanhActor(input_dim, action_dim).to(get_device())
        self.actor_target = MLPTanhActor(input_dim, action_dim).to(get_device())
        self.actor_target.eval()
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.Q = MLPCritic(input_dim, action_dim).to(get_device())
        self.Q_target = MLPCritic(input_dim, action_dim).to(get_device())
        self.Q_target.eval()  # we won't be passing gradients to this network
        self.Q_target.load_state_dict(self.Q.state_dict())

        # ===== optimizers =====

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        # ===== hyper-parameters =====

        self.gamma = gamma
        self.action_noise = action_noise
        self.polyak = polyak

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

        PREDICTIONS = self.Q(batch.s, batch.a)

        with torch.no_grad():

            na = self.actor_target(batch.ns)
            # oh my, this bug in the following line took me 2 days or so to find it
            # basically, if batch.mask has shape (64, ) and its multiplier has shape (64, 1)
            # the result is a (64, 64) tensor, but this does not even cause an error!!!
            TARGETS = batch.r + \
                  self.gamma * self.Q_target(batch.ns, na) * (1 - batch.d)

        Q_LEARNING_LOSS = torch.mean((PREDICTIONS - TARGETS.detach()) ** 2)

        # ==================================================
        # policy loss (not present in Q-learning)
        # ==================================================

        a = self.actor(batch.s)
        Q_VALUES = self.Q(batch.s, a)

        ACTOR_LOSS = - torch.mean(Q_VALUES)  # minimizing this loss is maximizing the q values

        # ==================================================
        # backpropagation and gradient descent
        # ==================================================

        self.actor_optimizer.zero_grad()
        ACTOR_LOSS.backward()  # inconveniently this back-props into prediction net as well, but (see following line)
        self.Q_optimizer.zero_grad()  # clear the gradient of the prediction net accumulated by ACTOR_LOSS.backward()
        Q_LEARNING_LOSS.backward()

        # doing a gradient clipping between -1 and 1 is equivalent to using Huber loss
        # guaranteed to improve stability so no harm in using at all
        self.clip_gradient(self.actor)
        self.clip_gradient(self.Q)

        self.Q_optimizer.step()
        self.actor_optimizer.step()

        self.polyak_update(old_net=self.actor_target, new_net=self.actor)
        self.polyak_update(old_net=self.Q_target, new_net=self.Q)

    def save_actor(self, save_dir: str) -> None:
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor.pth'))

    def load_actor(self, save_dir: str) -> None:
        self.actor.load_state_dict(torch.load(os.path.join(save_dir, 'actor.pth'), map_location=torch.device('cpu')))