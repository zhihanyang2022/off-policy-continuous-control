import os
import gin

from copy import deepcopy
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

        # hyper-parameters

        super().__init__(
            input_dim=input_dim,
            action_dim=action_dim,
            gamma=gamma,
            lr=lr,
            polyak=polyak
        )

        self.action_noise = action_noise

        # networks

        self.actor = MLPTanhActor(input_dim, action_dim).to(get_device())
        self.actor_targ = deepcopy(self.actor)
        set_requires_grad_flag(self.actor_targ, False)

        self.Q = MLPCritic(input_dim, action_dim).to(get_device())
        self.Q_targ = deepcopy(self.Q)
        set_requires_grad_flag(self.Q_targ, False)

        self.networks_dict.update({
            "actor": self.actor,
            "actor_targ": self.actor_targ,
            "Q": self.Q,
            "Q_targ": self.Q_targ
        })  # these networks will be saved as KEY.pth upon calls to super().save_networks

        # optimizers

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=lr)

    def act(self, state: np.array, deterministic: bool) -> np.array:
        with torch.no_grad():
            state = torch.tensor(state).unsqueeze(0).float().to(get_device())
            greedy_action = self.actor(state).view(-1).cpu().numpy()  # view as 1d -> to cpu -> to numpy
            if deterministic:
                return greedy_action
            else:
                return np.clip(greedy_action + self.action_noise * np.random.randn(len(greedy_action)), -1.0, 1.0)

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

        Q_loss = torch.mean((predictions - targets) ** 2)

        assert Q_loss.shape == ()

        # reduce td error

        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # compute policy loss

        a = self.actor(b.s)
        Q_values = self.Q(b.s, a)
        policy_loss = - torch.mean(Q_values)

        assert a.shape == (bs, self.action_dim)
        assert policy_loss.shape == ()

        # reduce policy loss

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks

        self.polyak_update(targ_net=self.actor_targ, pred_net=self.actor)
        self.polyak_update(targ_net=self.Q_targ, pred_net=self.Q)

        return {
            # for learning the q functions
            '(qfunc) Q pred': float(predictions.mean()),
            '(qfunc) Q loss': float(Q_loss),
            # for learning the actor
            '(actor) policy loss': float(policy_loss),
        }
