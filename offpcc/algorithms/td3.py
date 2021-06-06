import gin

import numpy as np
import torch
import torch.optim as optim

from basics.abstract_algorithms import OffPolicyRLAlgorithm
from basics.actors_and_critics import MLPTanhActor, MLPCritic
from basics.replay_buffer import Batch
from basics.utils import get_device, create_target, polyak_update, save_net, load_net


@gin.configurable(module=__name__)
class TD3(OffPolicyRLAlgorithm):

    def __init__(
        self,
        input_dim,
        action_dim,
        gamma=0.99,
        lr=3e-4,
        polyak=0.995,
        action_noise=0.1,  # standard deviation of action noise
        target_noise=0.2,  # standard deviation of target smoothing noise
        noise_clip=0.5,  # max abs value of target smoothing noise
        policy_delay=2
    ):

        # hyper-parameters

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.polyak = polyak

        self.action_noise = action_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        self.policy_delay = policy_delay

        # trackers

        self.num_Q_updates = 0  # for delaying updates
        self.mean_Q1_value = 0  # for logging; the actor does not get updated every iteration,
        # so this statistic is not available every iteration

        # networks

        self.actor = MLPTanhActor(input_dim, action_dim).to(get_device())
        self.actor_targ = create_target(self.actor)

        self.Q1 = MLPCritic(input_dim, action_dim).to(get_device())
        self.Q1_targ = create_target(self.Q1)

        self.Q2 = MLPCritic(input_dim, action_dim).to(get_device())
        self.Q2_targ = create_target(self.Q2)

        # optimizers

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=lr)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=lr)

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

        Q1_predictions = self.Q1(b.s, b.a)
        Q2_predictions = self.Q2(b.s, b.a)

        # compute targets

        with torch.no_grad():

            na = self.actor_targ(b.ns)
            noise = torch.clamp(
                torch.randn(na.size()) * self.target_noise, -self.noise_clip, self.noise_clip
            ).to(get_device())
            smoothed_na = torch.clamp(na + noise, -1, 1)

            n_min_Q_targ = torch.min(self.Q1_targ(b.ns, smoothed_na), self.Q2_targ(b.ns, smoothed_na))

            targets = b.r + self.gamma * (1 - b.d) * n_min_Q_targ

            assert na.shape == (bs, self.action_dim)
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

        self.num_Q_updates += 1

        if self.num_Q_updates % self.policy_delay == 0:  # delayed policy update; special in TD3

            # compute policy loss

            a = self.actor(b.s)
            Q1_values = self.Q1(b.s, a)  # val stands for values
            policy_loss = - torch.mean(Q1_values)

            self.mean_Q1_value = float(Q1_values.mean())
            assert a.shape == (bs, self.action_dim)
            assert Q1_values.shape == (bs, 1)
            assert policy_loss.shape == ()

            # reduce policy loss

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            # update target networks

            polyak_update(targ_net=self.actor_targ, pred_net=self.actor, polyak=self.polyak)
            polyak_update(targ_net=self.Q1_targ, pred_net=self.Q1, polyak=self.polyak)
            polyak_update(targ_net=self.Q2_targ, pred_net=self.Q2, polyak=self.polyak)

        return {
            # for learning the q functions
            '(qfunc) Q1 pred': float(Q1_predictions.mean()),
            '(qfunc) Q2 pred': float(Q2_predictions.mean()),
            '(qfunc) Q1 loss': float(Q1_loss),
            '(qfunc) Q2 loss': float(Q2_loss),
            # for learning the actor
            '(actor) Q1 value': self.mean_Q1_value
        }

    def save_actor(self, save_dir: str) -> None:
        save_net(net=self.actor, save_dir=save_dir, save_name="actor.pth")

    def load_actor(self, save_dir: str) -> None:
        load_net(net=self.actor, save_dir=save_dir, save_name="actor.pth")
