import gin

import numpy as np
import torch
import torch.optim as optim

from basics.abstract_algorithms import OffPolicyRLAlgorithm
from basics.actors_and_critics import MLPTanhActor, MLPCritic
from basics.replay_buffer import Batch
from basics.utils import get_device, create_target, polyak_update, save_net, load_net
from basics.lr_scheduler import LRScheduler


@gin.configurable(module=__name__)
class DDPG(OffPolicyRLAlgorithm):

    def __init__(
        self,
        input_dim,
        action_dim,
        gamma=0.99,
        lr=3e-4,
        lr_schedule=lambda num_updates: 1,
        polyak=0.995,
        action_noise=0.1,
    ):

        # hyper-parameters

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.polyak = polyak

        self.action_noise = action_noise

        # networks

        self.actor = MLPTanhActor(input_dim, action_dim).to(get_device())
        self.actor_targ = create_target(self.actor)

        self.Q = MLPCritic(input_dim, action_dim).to(get_device())
        self.Q_targ = create_target(self.Q)

        # optimizers

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        # lr scheduler

        self.lr_scheduler = LRScheduler(
            optimizers=[self.actor_optimizer, self.Q_optimizer],
            init_lr=lr,
            schedule=lr_schedule
        )

    def act(self, state: np.array, deterministic: bool) -> np.array:
        with torch.no_grad():
            state = torch.tensor(state).unsqueeze(0).float().to(get_device())
            greedy_action = self.actor(state).view(-1).cpu().numpy()  # view as 1d -> to cpu -> to numpy
            if deterministic:
                return greedy_action
            else:
                return np.clip(greedy_action + self.action_noise * np.random.randn(self.action_dim), -1.0, 1.0)

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

        polyak_update(targ_net=self.actor_targ, pred_net=self.actor, polyak=self.polyak)
        polyak_update(targ_net=self.Q_targ, pred_net=self.Q, polyak=self.polyak)

        # update learning rate

        self.lr_scheduler.update_lr()

        return {
            # for learning the q functions
            '(qfunc) Q pred': float(predictions.mean()),
            '(qfunc) Q loss': float(Q_loss),
            # for learning the actor
            '(actor) Q value': float(Q_values.mean()),
        }

    def save_actor(self, save_dir: str) -> None:
        save_net(net=self.actor, save_dir=save_dir, save_name="actor.pth")

    def load_actor(self, save_dir: str) -> None:
        load_net(net=self.actor, save_dir=save_dir, save_name="actor.pth")
